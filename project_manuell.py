#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:59:09 2019

@author: JanStrich
"""

import numpy as np
from skimage.io import imread, imsave
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import os
import glob
import Funktionsscript as my
import re

## Change Direction
os.chdir("/Users/JanStrich/Google Drive/Uni/Informatik/Computer Vision/")


##Trainingsdaten -----------------------------------------------------------------------------------------
#tr_listfiles = glob.glob('GroceryStoreDataset-V5/train/*.jpg')
#trImgs = []
#trLabels = []
#for t in range(len(tr_listfiles)):
#    trImgs.append(imread(tr_listfiles[t]))
#    trLabels.append(glob.glob('GroceryStoreDataset-V5/train/*.jpg')[t].split('/')[-1].split('.')[0])
#    
##Testdaten -----------------------------------------------------------------------------------------
#test_listfiles = glob.glob('GroceryStoreDataset-V5/test/*.jpg')
#testImgs = []
#testLabels = []
#for t in range(len(test_listfiles)):
#    testImgs.append(imread(test_listfiles[t]))
#    testLabels.append(glob.glob('GroceryStoreDataset-V5/test/*.jpg')[t].split('/')[-1].split('.')[0])
#
##Validierungsdaten -----------------------------------------------------------------------------------------
#vl_listfiles = glob.glob('GroceryStoreDataset-V5/val/*.jpg')
#vlImgs = []
#vlLabels = []
#for t in range(len(vl_listfiles)):
#    vlImgs.append(imread(vl_listfiles[t]))
#    vlLabels.append(glob.glob('GroceryStoreDataset-V5/val/*.jpg')[t].split('/')[-1].split('.')[0])
#        
## Labels umbenennen
#trLabels = label_function(trLabels)
#testLabels = label_function(testLabels)
#vlLabels = label_function(vlLabels)

trmeans = my.imgs_means(trImgs)  