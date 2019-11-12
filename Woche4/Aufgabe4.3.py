# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:40:19 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import glob
import Funktionsscript as my


## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche4\\haribo2')


## Trainingsdaten
tr_listfiles = glob.glob('./hariboTrain2/*.png')
trImgs = []
trLabels = []
for t in range(len(tr_listfiles)):
    trImgs.append(imread(tr_listfiles[t]))
    trLabels.append(glob.glob('./hariboTrain2/*.png')[t].split('\\')[-1].split('.')[0].split('_')[0])
 
## Validierungsdaten
vl_listfiles = glob.glob('./hariboVal2/*.png')
vlImgs = []
vlLabels = []
for t in range(len(vl_listfiles)):
    vlImgs.append(imread(vl_listfiles[t])) 
    vlLabels.append(glob.glob('./hariboVal2/*.png')[t].split('\\')[-1].split('.')[0].split('_')[0])
    
## 4.3.2 ----------------------------------------------------------------------------------------

trImgs_bb, vlImgs_bb = my.imgs_bb(trImgs, vlImgs, "otsu")
    
print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs_bb,vlImgs_bb, typ = "mean", verfahren = "euk")))
print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs_bb,vlImgs_bb, typ = "hist3d", verfahren = "euk")))

## 4.3.3 ----------------------------------------------------------------------------------------

## Hier soll das Verhältnis berechnet werden
## Dafür sollen wir nach irgeinem Maß die Bilder drehen. Ich weiß aber nicht nach welchem Kriterien
## ich den Winkel bestimmt nach dem ich das Bild drehen soll 

## 4.3.4 ----------------------------------------------------------------------------------------
print(my.imgs_hitrate(trLabels, vlLabels, 
      my.imgs_nearest_neighbor(trImgs_bb,vlImgs_bb, typ = "ecc", verfahren = "euk")))


