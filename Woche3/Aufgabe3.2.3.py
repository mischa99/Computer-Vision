# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:43:23 2019

@author: Jan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:51:30 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy import stats



## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche3')
data = np.load("trainingsDatenFarbe2.npz")
data2 = np.load("validierungsDatenFarbe2.npz")


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
   trmeans.append(np.mean(trImgs[t,:,:], axis = (0,1)))
   
vlmeans = []
for t in list(range(0,30,1)):
   vlmeans.append(np.mean(vlImgs[t,:,:], axis = (0,1)))


trhistred = []
for t in list(range(0,60,1)):
    trhistred.append(np.histogram(trImgs[t,:,0], bins = 256, range = (0,256)))

trhistyellow = []
for t in list(range(0,60,1)):
    trhistyellow.append(np.histogram(trImgs[t,:,1], bins = 256, range = (0,256)))

trhistblue = []
for t in list(range(0,60,1)):
    trhistblue.append(np.histogram(trImgs[t,:,2], bins = 256, range = (0,256)))

vlhistred = []
for t in list(range(0,30,1)):
    vlhistred.append(np.histogram(vlImgs[t,:,0], bins = 256, range = (0,256)))

vlhistyellow = []
for t in list(range(0,30,1)):
    vlhistyellow.append(np.histogram(vlImgs[t,:,1], bins = 256, range = (0,256)))

vlhistblue = []
for t in list(range(0,30,1)):
    vlhistblue.append(np.histogram(vlImgs[t,:,2], bins = 256, range = (0,256)))


hist = np.hstack([trhistred,trhistyellow,trhistblue])
#vlhist = []
#for t in list(range(0,30,1)):
#    vlhist.append(np.histogram(vlImgs[t,:,:], bins = 256, range = (0,256)))     
#   

##### Bilder vergleichen -----------------------------------------
   
    
best_blue = list(range(0,30))
temp = list(range(0,60))
for i in list(range(0,30,1)):
    for j in list(range(0,60,1)):
        temp[j] = euk_diff(vlhistblue[i],trhistblue[j])
    best_blue[i] = np.argmin(temp)
    
best_yellow = list(range(0,30))
temp = list(range(0,60))
for i in list(range(0,30,1)):
    for j in list(range(0,60,1)):
        temp[j] = euk_diff(vlhistyellow[i],trhistyellow[j])
    best_yellow[i] = np.argmin(temp)

best_red = list(range(0,30))
temp = list(range(0,60))
for i in list(range(0,30,1)):
    for j in list(range(0,60,1)):
        temp[j] = euk_diff(vlhistred[i],trhistred[j])
    best_red[i] = np.argmin(temp)
        
   
i = 0
kate_red = list(range(0,30))     
for t in best_red:
    kate_red[i] = trLabels[t]
    i+= 1
    
i = 0
kate_blue = list(range(0,30))     
for t in best_blue:
    kate_blue[i] = trLabels[t]
    i+= 1

i = 0
kate_yellow = list(range(0,30))     
for t in best_yellow:
    kate_yellow[i] = trLabels[t]
    i+= 1

best = list(range(0,30))
kate = list(range(0,30))
for t in kate:
    temp = [kate_red[t],kate_blue[t],kate_yellow[t]]
    best[t] = stats.mode(temp)[0][0]
    
    

Hitrate = sum(best == vlLabels)/len(kate)
