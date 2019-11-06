# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:55:52 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
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
    
## ------------------------------------------------------------------------------------------

trmeans = []
for t in list(range(0,len(listfiles))):
   trmeans.append(np.mean(trImgs[t], axis = (0,1)))
   
vlmeans = []
for t in list(range(0,len(listfiles2))):
   vlmeans.append(np.mean(vlImgs[t], axis = (0,1)))


best = list(range(0,len(listfiles2)))
temp = list(range(0,len(listfiles)))
for i in list(range(0,len(listfiles2))):
    for j in list(range(0,len(listfiles))):
            temp[j] = euk_diff(vlmeans[i],trmeans[j])
    best[i] = np.argmin(temp) 
   
i = 0
kate = list(range(0,len(listfiles2)))     
for t in best:
    kate[i] = trLabels[t]
    i+= 1


i = 0
for t in list(range(0,len(kate))):
    if (kate[t] == vlLabels[t]):
        i+= 1
Hitrate = i/len(kate)

##4.1.3 -----------------------------


trhist = []
for t in list(range(0,len(listfiles))):
    img = trImgs[t].reshape((trImgs[t].shape[0]* trImgs[t].shape[1],3))
    trhist.append(np.histogramdd(img, bins = [8,8,8], range = ((0,256),(0,256),(0,256)))[0])
    
vlhist = []
for t in list(range(0,len(listfiles2))):
    img = vlImgs[t].reshape((vlImgs[t].shape[0]* vlImgs[t].shape[1],3))
    vlhist.append(np.histogramdd(img, bins = [8,8,8], range = ((0,256),(0,256),(0,256)))[0])

best = list(range(0,len(listfiles2)))
temp = list(range(0,len(listfiles)))
for i in list(range(0,len(listfiles2))):
    for j in list(range(0,len(listfiles))):
            temp[j] = euk_diff(vlhist[i],trhist[j])
    best[i] = np.argmin(temp) 
   
i = 0
kate = list(range(0,len(listfiles2)))     
for t in best:
    kate[i] = trLabels[t]
    i+= 1


i = 0
for t in list(range(0,len(kate))):
    if (kate[t] == vlLabels[t]):
        i+= 1
Hitrate = i/len(kate)