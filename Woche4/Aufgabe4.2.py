# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:55:52 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import glob


####
def euk_diff(zahl1, zahl2):
    return np.sum((zahl1-zahl2)**2)**.5



## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche4\\haribo1')


## Trainingsdaten
listfiles = glob.glob('./hariboTrain/*.png')
trImgs = []
trLabels = []
for t in list(range(0,len(listfiles))):
    trImgs.append(imread(listfiles[t]))
    trLabels.append(glob.glob('./hariboTrain/*.png')[t].split('\\')[-1].split('.')[0].split('_')[0])
 
## Validierungsdaten
listfiles2 = glob.glob('./hariboVal/*.png')
vlImgs = []
vlLabels = []
for t in list(range(0,len(listfiles2))):
    vlImgs.append(imread(listfiles2[t])) 
    vlLabels.append(glob.glob('./hariboVal/*.png')[t].split('\\')[-1].split('.')[0].split('_')[0])


### -------------
trImgs_grey = []
for t in trImgs:
   trImgs_grey.append(t[:,:,0]/3 + t[:,:,1]/3 + t[:,:,2]/3)
vlImgs_grey = []
for t in vlImgs:
   vlImgs_grey.append(t[:,:,0]/3 + t[:,:,1]/3 + t[:,:,2]/3)

trImgs_mask = []
for t in trImgs_grey:
    trImgs_mask.append(t < 100)

vlImgs_mask = []
for t in vlImgs_grey:
    vlImgs_mask.append(t < 130)


trImgs_mask[0] = trImgs_mask[0].astype(np.int)
props = regionprops(trImgs_mask[0])[0]
plt.imshow(trImgs[0][props.bbox[0]:props.bbox[2],props.bbox[1]:props.bbox[3]], cmap="Greys_r")