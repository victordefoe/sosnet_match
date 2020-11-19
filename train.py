'''
Author: your name
Date: 2020-11-13 22:18:35
LastEditTime: 2020-11-19 12:05:26
LastEditors: Liu Chen
Description: Training process of sosnet. 
FilePath: \sosnet_match\train.py
  
'''

import torch
import torch.nn.functional as F
import numpy as np
from dataset import Train_Collection
from sosnet import SOSNet32x32
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='vislog')

N = 128
t = 1

def First_order(x, xplus):
    """
    shape: [N, 128]
    """
    def hard_sample_dis(a, b, i):
        m = float('+inf')
        for j in range(len(b)):
            if i != j:
                mj = min(F.pairwise_distance(a[i], a[j]),
                         F.pairwise_distance(a[i], a[j]),
                         F.pairwise_distance(b[i], a[j]),
                         F.pairwise_distance(b[i], b[j])
                         )
                m = min(m, mj)
        return m
    fos = torch.tensor([0]).float()
    for i in range(len(x)):
        di_pos = F.pairwise_distance(x[i], xplus[i], p=2)
        di_neg = hard_sample_dis(x, xplus, i)
        fos += (max(0, t+di_pos-di_neg)).pow(2)
    return fos/len(x)


def Second_order(x, xplus):
    sos = torch.tensor([0]).float()
    for i in range(len(x)):
        d2i = 0
        for j in range(len(xplus)):
            if i != j:
                d2i += (F.pairwise_distance(x[i], x[j]) -
                                 F.pairwise_distance(xplus[i], xplus[j])).pow(2)
        d2i = torch.sqrt(d2i)
        sos += d2i
    return sos/len(xplus)


Train_dataset = Train_Collection()
DS = DataLoader(Train_dataset, N)

if __name__ == '__main__':
    net = SOSNet32x32(dim_desc=128, drop_rate=0.1)
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-4)
    net.load_state_dict(torch.load('weights/sosnet_weight.pth'))

    itt = 66
    for epoch in range(4):
        for a, b in DS:
            x = net(a).unsqueeze(1)
            xplus = net(b).unsqueeze(1)

            fos = First_order(x, xplus)
            sos = Second_order(x, xplus)
            loss = fos + sos
            loss.backward()
            optimizer.step()
            print(float(loss.data), end=' ')
            writer.add_scalar('scalar/loss', float(loss.data), itt)
            itt += 1
            torch.save(net.state_dict(), 'weights/sosnet_weight.pth')

    writer.close()