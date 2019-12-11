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
from PIL import Image
from resizeimage import resizeimage


## Change Direction
os.chdir("/informatik2/students/home/8strich/Downloads/Project/GroceryStoreDataset-V5/")


# #Trainingsdaten -----------------------------------------------------------------------------------------
tr_listfiles = glob.glob('GroceryStoreDataset-V5_128/train/*.jpg')
trImgs = []
trLabels = []
for t in range(len(tr_listfiles)):
    trImgs.append(imread(tr_listfiles[t]))
    trLabels.append(glob.glob('GroceryStoreDataset-V5_128/train/*.jpg')[t].split('/')[-1].split('.')[0])
    
#Testdaten -----------------------------------------------------------------------------------------
#test_listfiles = glob.glob('GroceryStoreDataset-V5_128/test/*.jpg')
#testImgs = []
#testLabels = []
#for t in range(len(test_listfiles)):
#    testImgs.append(imread(test_listfiles[t]))
#    testLabels.append(glob.glob('GroceryStoreDataset-V5_128/test/*.jpg')[t].split('/')[-1].split('.')[0])

#Validierungsdaten -----------------------------------------------------------------------------------------
vl_listfiles = glob.glob('GroceryStoreDataset-V5_128/test/*.jpg')
vlImgs = []
vlLabels = []
for t in range(len(vl_listfiles)):
    vlImgs.append(imread(vl_listfiles[t]))
    vlLabels.append(glob.glob('GroceryStoreDataset-V5_128/test/*.jpg')[t].split('/')[-1].split('.')[0])
        
# Labels umbenennen
trLabels = my.imgs_labels(trLabels)
#testLabels = my.imgs_labels(testLabels)
vlLabels = my.imgs_labels(vlLabels)


#for i in test_listfiles:
#    os.chdir("/informatik2/students/home/8strich/Downloads/Project/GroceryStoreDataset-V5/")
#    img = open(i, 'rb')
#    img = Image.open(img)
#    img_new = resizeimage.resize_crop(img, [128,128])
#    os.chdir("/informatik2/students/home/8strich/Downloads/Project/GroceryStoreDataset-V5/GroceryStoreDataset-V5_128/test/")
#    img_new.save(i.split("/")[2], img.format)

# Diskriptoren erstellen -----------------------------------------------------------------------------------------
# mean
#tr_means = my.imgs_means(trImgs)
#vl_means = my.imgs_means(vlImgs)
#test_means = my.imgs_means(testImgs)
#
### sd
#tr_std = my.imgs_std(trImgs)
#vl_std = my.imgs_std(vlImgs)
#test_std = my.imgs_std(testImgs)

## hists
#tr_hist = my.imgs_hists(trImgs, bins = 32)
#vl_hist = my.imgs_hists(vlImgs, bins = 32)
##test_hist = my.imgs_hists(testImgs, bins = 32)
#
#tr_descriptors = np.array(tr_hist)
#vl_descriptors = np.array(vl_hist)


tr_hist = my.imgs_hists_3d(trImgs, bins = 16)
vl_hist = my.imgs_hists_3d(vlImgs, bins = 16)
#test_hist = my.imgs_hists_3d(testImgs, bins = 8)

#
#tr_descriptors = np.concatenate((tr_means,tr_std), axis=1)
#vl_descriptors = np.concatenate((vl_means,vl_std), axis=1)
#test_descriptors = np.concatenate((test_means,test_std), axis=1)


tr_descriptors = []
for t in tr_hist:
    tr_descriptors.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
vl_descriptors = []
for t in vl_hist:
    vl_descriptors.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))


tr_descriptors = np.array(tr_descriptors)
vl_descriptors = np.array(vl_descriptors)

tr_descriptors= tr_descriptors.astype('float32')
vl_descriptors= vl_descriptors.astype('float32')





## Klassifikation erstellen --------------------------------------------------------------------------------------------
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(tr_descriptors, trLabels)
predictions = clf.predict(vl_descriptors)


Hitrate = sum(predictions == vlLabels)
acc = sum(predictions == vlLabels)/len(vlLabels)

