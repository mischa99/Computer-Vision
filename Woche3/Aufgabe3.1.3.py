# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:34:12 2019

@author: Jan
"""


import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche3')
data = np.load("trainingsDaten2.npz")
data2 = np.load("validierungsDaten2.npz")


def euk_diff(hist1 , hist2):
    temp = hist1[0] - hist2[0]
    quad = 0
    for x in temp:
        quad+= x * x
    return np.sqrt(quad)



trImgs = data['data']
trLabels = data['labels']
vlImgs = data2['data']
vlLabels = data2['labels']



#### Merkmale berechnen ------------------------------------------
trmeans = []
for t in list(range(0,60,1)):
   trmeans.append(np.mean(trImgs[t,:,:]))
   
vlmeans = []
for t in list(range(0,30,1)):
   vlmeans.append(np.mean(vlImgs[t,:,:]))
   
trhist = []
for t in list(range(0,60,1)):
    trhist.append(np.histogram(trImgs[t,:,:], bins = 15, range = (0,256)))

vlhist = []
for t in list(range(0,30,1)):
    vlhist.append(np.histogram(vlImgs[t,:,:], bins = 15, range = (0,256)))
   
##### Bilder vergleichen -----------------------------------------
   
best = list(range(0,30))
temp = list(range(0,60))
for i in list(range(0,30,1)):
    for j in list(range(0,60,1)):
        temp[j] = euk_diff(vlhist[i],trhist[j])
    best[i] = np.argmin(temp)
        
   

kate = trLabels[best]

matrix = sklearn.metrics.confusion_matrix(vlLabels,kate)
print(matrix)     
        
        
