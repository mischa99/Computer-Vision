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



## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche3')
data = np.load("trainingsDatenFarbe2.npz")
data2 = np.load("validierungsDatenFarbe2.npz")


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
   
     
##### Bilder vergleichen -----------------------------------------
   
best = list(range(0,30,1))
for i in list(range(0,30,1)):
        temp = abs(trmeans - vlmeans[i])
        temp = np.mean(temp, axis = 1)
        best[i]  = np.argmin(temp)
   
kate = trLabels[best]


Hitrate = sum(kate == vlLabels)/len(kate)
