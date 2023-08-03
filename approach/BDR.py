import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BDR_loss(nn.Module):
    def __init__(self, m1=0.8, m2=0.8, momentum=0.99, tro=1.0):
        super(BDR_loss, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.momentum = momentum
        self.tro = tro

    def initialize(self, trn_loader, t, model):
        self.psi_k, self.psi_k_before_normalization = self.compute_psi(trn_loader)
        self.omega_k, self.mu_c_dict, self.label_freq_array, self.omega_k_dict = self.compute_omega(trn_loader, model)
        if t==0:
            self.pi = torch.zeros_like(self.psi_k).cuda()
        else:
            self.pi = self.m1 * self.psi_k + (1-self.m1) * self.omega_k
            self.pi = np.log(self.pi ** self.tro + 1e-12)
            self.pi = self.pi.cuda()
        return self.pi

    def momentum_update(self, features, targets):
        batch_label_freq_array = torch.zeros_like(self.psi_k)
        mu_c_dict = dict()
        label_freq = {}
        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :].clone().detach()
            else:
                mu_c_dict[y] += features[b, :].clone().detach()
            key = int(y)
            label_freq[key] = label_freq.get(key, 0) + 1

        assert len(torch.unique(targets)) == len(label_freq.keys()), print(torch.unique(targets), label_freq.keys())

        for i in label_freq.keys():
            mu_c_dict[i] /= torch.tensor(label_freq[i])
            weight1 = self.label_freq_array[i] / (self.label_freq_array[i] + label_freq[i])
            weight2 = label_freq[i] / (self.label_freq_array[i] + label_freq[i])
            self.mu_c_dict[i] =  weight1 * self.mu_c_dict[i] + weight2 * mu_c_dict[i]
            batch_label_freq_array[i] = label_freq[i]

        omega_k_dict = dict()
        for b in range(len(targets)):
            y = targets[b].item()
            if y not in omega_k_dict:
                omega_k_dict[y] = (features[b, :].clone().detach() - self.mu_c_dict[y]).unsqueeze(0) @ (features[b, :].clone().detach() - self.mu_c_dict[y]).unsqueeze(1)
            else:
                omega_k_dict[y] += (features[b, :].clone().detach() - self.mu_c_dict[y]).unsqueeze(0) @ (features[b, :].clone().detach() - self.mu_c_dict[y]).unsqueeze(1)

        for i in label_freq.keys():
            omega_k_dict[i] /= torch.tensor(label_freq[i])
            if label_freq[i] != 1:
                self.omega_k_dict[i] =  weight1 * self.omega_k_dict[i] + weight2 * omega_k_dict[i]

        pre_adjustments = []
        for kth in range(len(self.omega_k)):
            omega_k_dict_array = np.array(self.omega_k_dict[kth].cpu())
            pre_adjustments.append(float((1 / omega_k_dict_array).reshape(-1)))

        pre_adjustments = np.array(pre_adjustments)
        pre_adjustments = pre_adjustments / pre_adjustments.sum()
        self.omega_k = torch.from_numpy(pre_adjustments)

        hat_pi = self.m2 * self.psi_k + (1-self.m2) * self.omega_k
        hat_pi = np.log(hat_pi ** self.tro + 1e-12).cuda()

        self.pi = self.momentum * self.pi + (1 - self.momentum) * hat_pi
        return self.pi
    
    def compute_psi(self, train_loader, ):
        label_freq = {}
        for i, (inputs, target) in enumerate(train_loader):
            target = target.cuda()
            for j in target:
                key = int(j.item())
                label_freq[key] = label_freq.get(key, 0) + 1
        label_freq = dict(sorted(label_freq.items()))
        psi_k_before_normalization = np.array(list(label_freq.values()))
        psi_k = psi_k_before_normalization / psi_k_before_normalization.sum()
        psi_k = torch.from_numpy(psi_k)
        return psi_k, psi_k_before_normalization
    
    def compute_omega(self, train_loader, model):
        """compute the base probabilities"""
        _, fk_dict, label_freq_array = self.compute_fk(model, train_loader)
        _, omega_k_dict = self.compute_omega_k(model, fk_dict, train_loader, label_freq_array)
        K = len(fk_dict)
        pre_adjustments = []

        for kth in range(K):
            omega_k_dict_array = np.array(omega_k_dict[kth].cpu())
            pre_adjustments.append(float((1 / omega_k_dict_array).reshape(-1)))

        pre_adjustments = np.array(pre_adjustments)

        pre_adjustments = pre_adjustments / pre_adjustments.sum()
        omega = torch.from_numpy(pre_adjustments)
        return omega, fk_dict, label_freq_array, omega_k_dict

    def compute_fk(self, model, dataloader):
        mu_G = 0 # global feature center 【all classes】
        mu_c_dict = dict()  # local feature center 【each class】
        label_freq = {}
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs, features = model(inputs, return_features=True)

            mu_G += torch.sum(features, dim=0)

            for b in range(len(targets)):
                y = targets[b].item()
                if y not in mu_c_dict:
                    mu_c_dict[y] = features[b, :]
                else:
                    mu_c_dict[y] += features[b, :]

                key = int(y)
                label_freq[key] = label_freq.get(key, 0) + 1

        label_freq = dict(sorted(label_freq.items()))
        label_freq_array = np.array(list(label_freq.values()))

        mu_G /= sum(label_freq_array)
        for i in range(len(label_freq_array)):
            mu_c_dict[i] /= torch.tensor(label_freq_array[i])

        return mu_G, mu_c_dict, label_freq_array

    def compute_omega_k(self, model, mu_c_dict, train_loader, label_freq_array):
        # within-class covariance
        omega_k = 0
        omega_k_dict = dict()
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                outputs, features = model(inputs, return_features=True)

            for b in range(len(targets)):
                y = targets[b].item()
                omega_k += (features[b, :] - mu_c_dict[y]).unsqueeze(0) @ (features[b, :] - mu_c_dict[y]).unsqueeze(1)
                if y not in omega_k_dict:
                    omega_k_dict[y] = (features[b, :] - mu_c_dict[y]).unsqueeze(0) @ (features[b, :] - mu_c_dict[y]).unsqueeze(1)
                else:
                    omega_k_dict[y] += (features[b, :] - mu_c_dict[y]).unsqueeze(0) @ (features[b, :] - mu_c_dict[y]).unsqueeze(1)

        omega_k /= sum(label_freq_array)
        for i in range(len(label_freq_array)):
            omega_k_dict[i] /= torch.tensor(label_freq_array[i])

        return omega_k.cpu().numpy(), omega_k_dict 

