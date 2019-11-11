# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:55:52 2019

@author: Jan
"""

import os
from skimage.io import imread, imsave
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
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


### 4.2.1 -----------------------------------------------------------------------------------

trImgs_bb, vlImgs_bb = my.imgs_bb(trImgs, vlImgs, 100)

#4.2.2 ---------------------------------------------------------------------------------------

print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs_bb,vlImgs_bb, typ = "mean", verfahren = "euk")))    
print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs_bb,vlImgs_bb, typ = "hist3d", verfahren = "euk")))

# 4.2.3 -------------------------------------------------------------------------------------

trImgs_bb, vlImgs_bb = my.imgs_bb(trImgs, vlImgs, Schwellenwert= "otsu")
print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs_bb,vlImgs_bb, typ = "hist3d", verfahren = "euk")))

