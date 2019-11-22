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


from skimage.io import imread, imsave
import matplotlib.pyplot as plt

#always do this before working with keras!
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

from tensorflow import set_random_seed
set_random_seed(123)# -''-

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD



## Change Direction

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

trstds = []
for t in list(range(0,60,1)):
   trstds.append(np.std(trImgs[t,:,:], axis = (0,1)))
   
vlstds = []
for t in list(range(0,30,1)):
   vlstds.append(np.std(vlImgs[t,:,:], axis = (0,1)))   

tr_descriptors = np.concatenate((trmeans,trstds), axis=1) #axis=1 -> column wise, else row
vl_descriptors = np.concatenate((vlmeans,vlstds), axis=1)

tr_descriptors=tr_descriptors.astype('float32')
vl_descriptors=vl_descriptors.astype('float32')
#print(tr_descriptors.dtype)   #how to check if successfull

arr=[1,4,8]
y=0
for x in arr:
    trLabels=np.where(trLabels==x,y,trLabels)
    vlLabels=np.where(vlLabels==x,y,vlLabels)
    y+=1


##### Bilder vergleichen -----------------------------------------
   
best = list(range(0,30,1))
for i in list(range(0,30,1)):
        temp = abs(trmeans - vlmeans[i])
        temp = np.mean(temp, axis = 1)
        best[i]  = np.argmin(temp)
   
kate = trLabels[best]


Hitrate = sum(kate == vlLabels)/len(kate)