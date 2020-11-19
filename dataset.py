'''
Author: your name
Date: 2020-11-13 16:38:22
LastEditTime: 2020-11-18 14:15:13
LastEditors: Liu Chen
Description: 
FilePath: \sosnet_match\dataset.py
  
'''

import os
import json
import random
import pandas as pd
import numpy as np
from os.path import join as opj
import torch
import torchvision
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageEnhance, ImageFilter
import skimage
from skimage import util

# modify the config.py in different enviornment for path configuration
from config import *
training_dataset_root = TRAIN_DATA_ROOT
train_gt = TRAIN_GT_PATH


# training data
class Train_Collection(Dataset):
    def __init__(self, stage='val', tun_pam=[0.5], img_size=(32,32)):
        sourceA = opj(training_dataset_root,'patch')
        sourceB = opj(training_dataset_root,'seps','18')
        dists_path = opj(training_dataset_root,'dists_forms')
        gts = json.load(open(train_gt))

        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:] == 'bmp']
        self.B_imgs = [opj(sourceB, i)
                       for i in gts.values() if i[-3:] == 'jpg']

        self.stage = stage
        prepro = []
        prepro.append(transforms.Resize(size=img_size))
        prepro.append(transforms.ToTensor())
        self.trans = transforms.Compose(prepro)
        self.tun_pam = tun_pam[0]
        # print(self.tun_pam)

    def __len__(self):
        return len(self.A_imgs)
            
    def __getitem__(self, item):
        
        imgA = Image.open(self.A_imgs[item])
        imgB = Image.open(self.B_imgs[item])
        img_A = self.trans(imgA)
        img_B = self.trans(imgB)

        return img_A, img_B


    



        


