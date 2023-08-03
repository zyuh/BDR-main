import torch
from torch import nn
from copy import deepcopy

import math
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import normal
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        cosine = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        #out = x.mm(self.weight)
        return cosine

class NoiseLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NoiseLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.simpler = normal.Normal(0, 1/3)
        self.in_features = in_features
        self.out_features = out_features
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #限制噪声幅度在[-1,1]
        noise  = torch.randn_like(input) #noise = self.simpler.sample(input.shape).clamp(-1, 1).to(input.device)#
        
        out = F.linear(input, self.weight, self.bias)
        noise = F.linear(noise, self.weight)
        # output = [out,noise]
        output = out + noise
        return output


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs, bias=False))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def add_head_v2(self, num_outputs, use_norm, use_noise):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        if use_norm:
            self.heads.append(NormedLinear(self.out_size, num_outputs))
        elif use_noise:
            self.heads.append(NoiseLinear(self.out_size, num_outputs))
        else:
            self.heads.append(nn.Linear(self.out_size, num_outputs, bias=False))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def add_head_v3(self, num_outputs, init_weight, t):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        head = nn.Linear(self.out_size, num_outputs, bias=False)
        if t == 0:
            head.weight = torch.nn.Parameter(init_weight[t:t+num_outputs, :], requires_grad=False)
            self.classes_num_until_now = num_outputs
        else:
            head.weight = torch.nn.Parameter(init_weight[self.classes_num_until_now:self.classes_num_until_now+num_outputs, :], requires_grad=False)
            self.classes_num_until_now += num_outputs
        self.heads.append(head)
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    # hard coded a interface specifically for podnet
    # output: prediction y, features x, pod_features
    def forward_pod(self, x):
        x, pod_features = self.model(x, return_pod=True)
        y = []
        for head in self.heads:
            y.append(head(x))
        return y, x, pod_features

    def forward_repres(self, x):
        repres = self.model(x)
        return repres

    def forward_cls(self, repres):
        y = []
        for head in self.heads:
            y.append(head(repres))
        return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
