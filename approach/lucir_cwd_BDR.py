import copy
import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from dataset.exemplars_dataset import ExemplarsDataset

from torch.nn.parallel import DistributedDataParallel as DDP
from .lucir_utils import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
import time
# ----------------------- auxiliary loss specifics -----------------------------
from .aux_loss import DecorrelateLossClass
from .utils import reduce_tensor_mean, reduce_tensor_sum, global_gather
# ------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as scilin
from torch.distributions import normal
from .BDR import BDR_loss
import os

class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """
    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, decay_mile_stone=[80,120], lr_decay=0.1, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, ddp=False, local_rank=0, logger=None, exemplars_dataset=None,
                 lamb=5., lamb_mr=1., dist=0.5, K=2, scale=1.0,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False,
                 # aux specifics
                 cwd=True, aux_coef=0.1, reject_threshold=1, first_task_lr=0.1, first_task_bz=128,
                 # -----------------------------------------------
                 BDR=False, m1=0.0, m2=0.0, momentum_update=0.99, tro=1.0,
                 ):
        super(Appr, self).__init__(model, device, nepochs, lr, decay_mile_stone, lr_decay, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, ddp, local_rank,
                                   logger, exemplars_dataset)
        self.lamb = lamb
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.scale = scale
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamb
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss

        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

        # ---------------- auxiliary loss specifics -------------------
        self.aux_coef = aux_coef
        self.reject_threshold = reject_threshold
        self.first_task_lr = first_task_lr
        self.first_task_bz = first_task_bz

        self.aux_loss = DecorrelateLossClass(reject_threshold=self.reject_threshold, ddp=ddp)
        # -------------------------------------------------------------
        # flag variable that identify whether we're learning the first task
        self.first_task = True
        self.cwd = cwd
        
        self.BDR = BDR
        self.m1 = m1
        self.m2 = m2
        self.momentum_update = momentum_update
        self.tro = tro
        self.BDR_dynamic = BDR_loss(self.m1, self.m2, self.momentum_update, self.tro)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamb', default=5., type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamb-mr', default=1., type=float, required=False,
                            help='Trade-off for the MR loss (default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')
        # Flags for ablating the approach
        parser.add_argument('--remove-less-forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove-margin-ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')

        # ------------------ auxiliary loss specifics -------------------
        parser.add_argument('--aux-coef', default=0.1, type=float, required=False,
                            help='coefficient for auxiliary loss')
        parser.add_argument('--reject-threshold', default=1, type=int, required=True,
                            help='rejection threshold for calculating correlation')
        parser.add_argument('--first-task-lr', default=0.1, type=float)
        parser.add_argument('--first-task-bz', default=128, type=int)
        # ---------------------------------------------------------------

        _group = parser.add_mutually_exclusive_group()
        # ---------------------------------------------------------------
        parser.add_argument('--cwd', action='store_true', required=False)
        parser.add_argument('--BDR', action='store_true', required=False)
        parser.add_argument('--tro', default=1.0, type=float)
        parser.add_argument('--m1', default=0.7, type=float)
        parser.add_argument('--m2', default=0.8, type=float)
        parser.add_argument('--momentum_update', default=0.99, type=float)
        return parser.parse_known_args(args)
        

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code)
            params = list(model.model.parameters()) + list(model.heads[-1].parameters())
        else:
            params = model.parameters()

        if self.first_task:
            self.first_task = False
            optimizer = torch.optim.SGD(params, lr=self.first_task_lr, weight_decay=self.wd, momentum=self.momentum)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        print(optimizer.param_groups[0]['lr'])
        return optimizer


    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if model.model.__class__.__name__ == 'ResNetCifar':
                old_block = model.model.layer3[-1]
                model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNet':
                try:
                    old_block = model.model.layer4[-1]
                    model.model.layer4[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                                old_block.conv2, old_block.bn2, old_block.downsample)
                except:
                    old_block = model.model.layer3[-1]
                    model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                                old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNetBottleneck':
                old_block = model.model.layer4[-1]
                model.model.layer4[-1] = BottleneckNoRelu(old_block.conv1, old_block.bn1,
                                                          old_block.relu, old_block.conv2, old_block.bn2,
                                                          old_block.conv3, old_block.bn3, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")

        # Changes the new head to a CosineLinear
        model.heads[-1] = CosineLinear(model.heads[-1].in_features, model.heads[-1].out_features)
        model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            model.heads[-1].sigma = model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in model.heads[:-1]])
                                                   / model.heads[-1].out_features)
                if self.local_rank == 0:
                    print('lambda value after adaptation: ', self.lamda)

        # if ddp option is activated, need to re-wrap the ddp model
        if self.ddp:
            self.model = DDP(self.model.module, device_ids=[self.local_rank])
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)


    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if t == 0:
            dset = trn_loader.dataset
            trn_loader = torch.utils.data.DataLoader(dset,
                    batch_size=self.first_task_bz,
                    sampler=trn_loader.sampler,
                    num_workers=trn_loader.num_workers,
                    pin_memory=trn_loader.pin_memory)

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            dset = trn_loader.dataset + self.exemplars_dataset
            if self.ddp:
                trn_sampler = torch.utils.data.DistributedSampler(dset, shuffle=True)
                trn_loader = torch.utils.data.DataLoader(dset,
                                                         batch_size=trn_loader.batch_size,
                                                         sampler=trn_sampler,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory)
            else:
                trn_loader = torch.utils.data.DataLoader(dset,
                                                         batch_size=trn_loader.batch_size,
                                                         shuffle=True,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory)

        #############################
        self.pi = self.BDR_dynamic.initialize(trn_loader, t, self.model)
        #############################

        # FINETUNING TRAINING -- contains the epochs loop
        # super().train_loop(t, trn_loader, val_loader)
        self.optimizer = self._get_optimizer()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay_mile_stone, gamma=self.lr_decay)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, e)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader, e)
                clock2 = time.time()
                if self.local_rank == 0:
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                if self.local_rank == 0:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader, e)
            clock4 = time.time()
            if self.local_rank == 0:
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            scheduler.step()
            print()


    def post_train_process(self, t, trn_loader, val_loader):
        # select exemplars
        if len(self.exemplars_dataset) > 0 and t > 0:
            dset = trn_loader.dataset + self.exemplars_dataset
        else:
            dset = trn_loader.dataset
        trn_loader = torch.utils.data.DataLoader(dset,
            batch_size=trn_loader.batch_size, shuffle=False, num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory)


        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform, self.ddp)

        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        if self.ddp:
            for h in self.ref_model.module.heads:
                h.train()
            self.ref_model.module.freeze_all()
        else:
            for h in self.ref_model.heads:
                h.train()
            self.ref_model.freeze_all()

    def train_epoch(self, t, trn_loader, e):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            # Forward previous model
            ref_outputs = None
            ref_features = None
            if t > 0:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
            loss_lucir = self.criterion(t, outputs, targets, ref_outputs, features, ref_features, e)

            # --------------- auxiliary loss specifics --------------------
            if t == 0 and self.cwd:
                features = F.normalize(features, p=2, dim=-1)
                loss_sc = self.aux_loss(features, targets)
            else:
                loss_sc = 0.0
            # -------------------------------------------------------------

            loss = loss_lucir + self.aux_coef*loss_sc

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None, e=None):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None:
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).to(self.device).view(-1, 1))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            # --------------- logit adjustment specifics --------------------
            if self.BDR:
                # each_session_classes_num_list = torch.tensor([int(o['wsigma'].shape[1]) for o in outputs])
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
                if t > 0:
                    #############################
                    self.pi = self.BDR_dynamic.momentum_update(features, targets)
                    #############################
                loss_ce = nn.CrossEntropyLoss()(outputs + self.pi, targets)
            else:
                loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)

            # Eq. 9: integrated objective
            loss = loss_dist + loss_ce + loss_mr

        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        if self.ddp:
            for m in range(len(pred)):
                this_task = (self.model.module.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + self.model.module.task_offset[this_task]
        else:
            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def eval(self, t, val_loader, e=None):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), e)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)

                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
