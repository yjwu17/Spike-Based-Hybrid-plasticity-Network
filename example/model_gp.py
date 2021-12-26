import torch,time,os
import torch.nn as nn
import torch.nn.functional as F
import math
from shared_parameters  import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import random
import  matplotlib.pyplot as plt
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ActFun(torch.autograd.Function):
    '''
    Approaximation function of spike firing rate function
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

# Network structure
cfg_fc = [512, 10]

probs = 0.0 # dropout rate
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self ):
        super(SNN_Model, self).__init__()

        self.fc1 = nn.Linear(28*28, cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )


        self.alpha1 = torch.nn.Parameter((1e-3 * torch.rand(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-3 * torch.rand(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((1e-3* torch.rand(1,cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-3 * torch.rand(1,cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.rand(1, 784)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)


    def mask_weight(self):
        self.fc1.weight.data = self.fc1.weight.data * self.mask1
        self.fc2.weight.data = self.fc2.weight.data * self.mask2

    def produce_hebb(self):
        hebb1 = torch.zeros(784, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

    def parameter_split(self):
        '''
        Split the meta-local parameters and gp-based parameters for different update methods
        '''
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'fv':
                base_param.append(p)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param


    def forward(self, input,hebb1, hebb2, wins = time_window):

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        # hebb1 = torch.zeros(784, cfg_fc[0], device=device)

        for step in range(wins):

            decay_factor = np.exp(- step / tau_w)

            x = input
            # x = input > torch.rand(input.size(), device = device) # generate spike trains
            x = x.view(batch_size, -1).float()

            h1_mem, h1_spike, hebb1= mem_update_nonplastic(self.fc1,  self.alpha1, self.beta1, self.gamma1,self.eta1,
                                          x*decay_factor, h1_spike, h1_mem, hebb1)

            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike, hebb2 = mem_update_nonplastic(self.fc2,  self.alpha2,  self.beta2, self.gamma2,self.eta2,
                                          h1_spike*decay_factor, h2_spike, h2_mem, hebb2)

            h2_sumspike = h2_sumspike + h2_spike

        outs = h2_mem/thresh

        return outs.clamp(max = 1.1), h1_sumspike, h2_sumspike, hebb1.data, hebb2.data, self.eta1, self.eta2


def mem_update_nonplastic(fc, alpha, beta, gamma, eta, inputs,  spike, mem, hebb):
    '''
    Update the membrane potentials
    Note that : The only difference between the GP and HP model is whether to use hebb-based local variables
    :param fc: linear opetrations
    :param alpha: the weight of hebb module
    :param beta: the meta-local parameters to control the learning rate
    :param gamma: the meta-local parameters to control the weight decay, not used in this demo
    :param eta: the meta-local parameters  of sliding threshold
    :return: current membrane potentials, spikes, and local states
    '''
    state = fc(inputs)
    mem = (mem - spike * thresh) * decay + state
    now_spike = act_fun(mem - thresh)
    # Update local modules
    # hebb = w_decay * hebb + torch.bmm((inputs * beta).unsqueeze(2), ((mem/thresh) - eta).tanh().unsqueeze(1)).mean(dim=0).squeeze()
    # hebb = hebb.clamp(min = -4, max = 4)
    return mem, now_spike.float(), hebb


