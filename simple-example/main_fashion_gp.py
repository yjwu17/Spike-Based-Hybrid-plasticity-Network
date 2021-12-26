# -*- coding: utf-8 -*-
"""
Created on Sat Dec 1 2019

A simple example of Hybrid plasticity SNN on fashion-MNIST datasets

Python 3.5.2

"""

from __future__ import print_function
import sys
import torchvision
import torchvision.transforms as transforms
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import time
from model_gp import *
import argparse
from shared_parameters  import *

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=r'../data/fashion/')
parser.add_argument('--f', type=str, default='../Checkp-fashion/baseline')
parser.add_argument('--names', type=str, default='mlp_v0')

opt = parser.parse_args()

data_path = opt.p
save_path = opt.f
names = opt.names
lambdas = 0.

train_dataset = torchvision.datasets.FashionMNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.FashionMNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0.  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
spi_record = list([])
loss_train_record = list([])
loss_test_record = list([])

criterion = nn.MSELoss()

total_best_acc = []
total_acc_record = []
total_hid_state = []

list_alpha1 = []
list_alpha2 = []
list_beta1 = []
list_beta2 = []
list_eta1 = []
list_eta2 = []

exp_num = 1
for exp in range(exp_num):
    setup_seed(111)
    snn = SNN_Model()
    snn.to(device)
    # optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
    param_base, param_local = snn.parameter_split()
    optim_base = torch.optim.Adam(param_base, lr=1e-3)
    optim_local = torch.optim.Adam(param_local, lr=5e-4)

    acc_record = []
    hebb1 , hebb2 = snn.produce_hebb()
    for epoch in range(num_epochs):
        running_loss = 0.
        snn.train()
        start_time = time.time()

        total = 0.
        correct = 0.

        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            images = images.float().to(device)

            for i_update in range(num_updates):
                optim_base.zero_grad()
                outputs, spikes, _, hebb1, hebb2, eta1, eta2 = snn(input=images,hebb1 = hebb1, hebb2 = hebb2, wins = time_window)
                labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)

                loss_reg = torch.norm(eta1, p = 2) + torch.norm(eta2, p = 2)
                loss = criterion(outputs.cpu(), labels_) + lambdas * loss_reg.cpu()
                loss.backward()
                optim_base.step()
                if i_update < (num_updates - 1): optim_local.zero_grad()

            optim_local.step()
            optim_local.zero_grad()
            running_loss += loss.item()


        print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'%(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
        print('Runing time:', time.time() - start_time)
        start_time = time.time()
        correct = 0.
        total = 0.
        running_loss = 0.

        # optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
        snn.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                # optimizer.zero_grad()
                outputs, sumspike,_, _, _, _, _ = snn(input=inputs, hebb1=hebb1, hebb2=hebb2, wins=time_window)

                labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)

                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))

                correct += float(predicted.eq(targets).sum().item())

            acc = 100. * float(correct) / float(total)

        print('Iters:', epoch,'\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        spi_record.append(sumspike.mean().detach())

    # Robustness Exp.
    import skimage
    for i in range(7):
        correct = 0.
        total = 0.
        Spike = 0.
        Spike2 = 0.
        noise = 'gaussian'
        with torch.no_grad():
            va = i * 0.01
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # 'gaussian','salt','pepper','s&p','speckle'
                if noise == 'gaussian':
                    inputsf = skimage.util.random_noise(inputs, mode='gaussian', clip=True, mean=0, var=1 * va)
                elif noise == 'salt':
                    inputsf = skimage.util.random_noise(inputs, mode='salt', clip=True, amount=va)
                elif noise == 'pepper':
                    inputsf = skimage.util.random_noise(inputs, mode='pepper', clip=True, amount=6 * va)
                elif noise == 'sp':
                    inputsf = skimage.util.random_noise(inputs, mode='s&p', clip=True, amount=2 * va)
                elif noise == 'speckle':
                    inputsf = skimage.util.random_noise(inputs, mode='speckle', clip=True, mean=0, var=10000 * va)

                inputsf = torch.from_numpy(inputsf)
                inputsf = inputsf.to(device)

                outputs, sumspike, sumspike2, _, _, _, _ = snn(input=inputsf, hebb1=hebb1, hebb2=hebb2,
                                                               wins=time_window)

                targets_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), targets_)

                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                Spike += sumspike.detach() / 8.0
                Spike2 += sumspike2.detach() / 8.0

            acc = 100. * float(correct) / float(total)
            Spikem = Spike.mean().cpu().numpy()
            Spikem2 = Spike2.mean().cpu().numpy()

        # print('Model:', names)
        print('{%s}||noise{%s}-i: %d || Test Accuracy : %.3f || Spike1:  %.3f || Spike2:  %.3f' % (
        names, noise, i, acc, Spikem, Spikem2))



    print('Complete..')

    print('Complete..')


    print('Complete..')




