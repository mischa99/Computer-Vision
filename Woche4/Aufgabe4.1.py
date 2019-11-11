# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:55:52 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
import glob
import Funktionsscript as my


## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche4\\haribo1')

## Trainingsdaten
tr_listfiles = glob.glob('./hariboTrain/*.png')
trImgs = []
trLabels = []
for t in range(len(tr_listfiles)):
    trImgs.append(imread(tr_listfiles[t]))
    trLabels.append(glob.glob('./hariboTrain/*.png')[t].split('\\')[-1].split('.')[0].split('_')[0])
 
## Validierungsdaten
vl_listfiles = glob.glob('./hariboVal/*.png')
vlImgs = []
vlLabels = []
for t in range(len(vl_listfiles)):
    vlImgs.append(imread(vl_listfiles[t])) 
    vlLabels.append(glob.glob('./hariboVal/*.png')[t].split('\\')[-1].split('.')[0].split('_')[0])
    
## 4.1.2  ------------------------------------------------------------------------------------------

print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs,vlImgs, typ = "mean", verfahren = "euk")))


##4.1.3 -------------------------------------------------------------------------------------------

print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs,vlImgs, typ = "hist3d", verfahren = "euk")))
