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
from sklearn.neighbors import KNeighborsClassifier

## Change Direction
os.chdir("/Users/JanStrich/Google Drive/Uni/Informatik/Computer Vision/")


# #Trainingsdaten -----------------------------------------------------------------------------------------
tr_listfiles = glob.glob('GroceryStoreDataset-V5/train/*.jpg')
trImgs = []
trLabels = []
for t in range(len(tr_listfiles)):
    trImgs.append(imread(tr_listfiles[t]))
    trLabels.append(glob.glob('GroceryStoreDataset-V5/train/*.jpg')[t].split('/')[-1].split('.')[0])
    
#Testdaten -----------------------------------------------------------------------------------------
test_listfiles = glob.glob('GroceryStoreDataset-V5/test/*.jpg')
testImgs = []
testLabels = []
for t in range(len(test_listfiles)):
    testImgs.append(imread(test_listfiles[t]))
    testLabels.append(glob.glob('GroceryStoreDataset-V5/test/*.jpg')[t].split('/')[-1].split('.')[0])

#Validierungsdaten -----------------------------------------------------------------------------------------
vl_listfiles = glob.glob('GroceryStoreDataset-V5/val/*.jpg')
vlImgs = []
vlLabels = []
for t in range(len(vl_listfiles)):
    vlImgs.append(imread(vl_listfiles[t]))
    vlLabels.append(glob.glob('GroceryStoreDataset-V5/val/*.jpg')[t].split('/')[-1].split('.')[0])
        
# Labels umbenennen
trLabels = my.imgs_labels(trLabels)
testLabels = my.imgs_labels(testLabels)
vlLabels = my.imgs_labels(vlLabels)



## Diskriptoren erstellen -----------------------------------------------------------------------------------------
## mean
tr_means = my.imgs_means(trImgs)
vl_means = my.imgs_means(vlImgs)
test_means = my.imgs_means(testImgs)

## sd
tr_std = my.imgs_std(trImgs)
vl_std = my.imgs_std(vlImgs)
test_std = my.imgs_std(testImgs)

## hists
tr_hist = my.imgs_hists(trImgs)
vl_hist = my.imgs_hists(vlImgs)
test_hist = my.imgs_hists(testImgs)


tr_descriptors = np.concatenate((tr_means,tr_std,tr_hist), axis=1)
vl_descriptors = np.concatenate((vl_means,vl_std,vl_hist), axis=1)
test_descriptors = np.concatenate((test_means,test_std, test_hist), axis=1)
tr_descriptors= tr_descriptors.astype('float32')
vl_descriptors= vl_descriptors.astype('float32')
test_descriptors= test_descriptors.astype('float32')





## Klassifikation erstellen --------------------------------------------------------------------------------------------
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(tr_descriptors, trLabels)
predictions = clf.predict(vl_descriptors)


Hitrate = 0
for i in range(len(vlLabels)):
    if predictions[i] == vlLabels[i]:
        Hitrate+= 1
acc = Hitrate/len(vlLabels)

