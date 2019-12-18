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
import re
from skimage.color import rgb2hsv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage.feature import hog

## Change Direction
#os.chdir("/informatik2/students/home/8strich/Downloads/Project/GroceryStoreDataset-V5/")
os.chdir("/Users/JanStrich/Google Drive/Uni/Informatik/Computer Vision/")

 #Trainingsdaten -----------------------------------------------------------------------------------------
tr_listfiles = glob.glob('GroceryStoreDataset-V5_128/train/*.jpg')
trImgs = []
trLabels = []
for t in range(len(tr_listfiles)):
    trImgs.append(imread(tr_listfiles[t]))
    trLabels.append(glob.glob('GroceryStoreDataset-V5_128/train/*.jpg')[t].split('/')[-1].split('.')[0])

#Validierungsdaten -----------------------------------------------------------------------------------------
vl_listfiles = glob.glob('GroceryStoreDataset-V5_128/val/*.jpg')
vlImgs = []
vlLabels = []
for t in range(len(vl_listfiles)):
    vlImgs.append(imread(vl_listfiles[t]))
    vlLabels.append(glob.glob('GroceryStoreDataset-V5_128/val/*.jpg')[t].split('/')[-1].split('.')[0])
    
##Testdaten -----------------------------------------------------------------------------------------
##test_listfiles = glob.glob('GroceryStoreDataset-V5_128/test/*.jpg')
##testImgs = []
##testLabels = []
##for t in range(len(test_listfiles)):
##    testImgs.append(imread(test_listfiles[t]))
##    testLabels.append(glob.glob('GroceryStoreDataset-V5_128/test/*.jpg')[t].split('/')[-1].split('.')[0])
#
#hsv_trImgs = []
#for t in trImgs:
#     hsv_trImgs.append(rgb2hsv(t))
#trImgs = hsv_trImgs
#hsv_vlImgs = []
#for t in vlImgs:
#     hsv_vlImgs.append(rgb2hsv(t))
#vlImgs = hsv_vlImgs
#
#
#        
# Labels umbenennen
trLabels = my.imgs_labels(trLabels)
vlLabels = my.imgs_labels(vlLabels)


## Bilder zentrieren ------------------------------------------------------------------------------------------------
#for i in test_listfiles:
#    os.chdir("/informatik2/students/home/8strich/Downloads/Project/GroceryStoreDataset-V5/")
#    img = open(i, 'rb')
#    img = Image.open(img)
#    img_new = resizeimage.resize_crop(img, [128,128])
#    os.chdir("/informatik2/students/home/8strich/Downloads/Project/GroceryStoreDataset-V5/GroceryStoreDataset-V5_128/test/")
#    img_new.save(i.split("/")[2], img.format)


#hog_trImgs = []
#for t in trImgs:
#    hog_trImgs.append(hog(t, orientations=16, pixels_per_cell=(16,16), cells_per_block=(1, 1), visualize=True, multichannel=True))
#hog_trImgs = []
#for t in trImgs:
#    hog_trImgs.append(hog(t, orientations=16, pixels_per_cell=(16,16), cells_per_block=(1, 1), visualize=True, multichannel=True))

tr_3d = my.imgs_hists_3d(trImgs, bins = 8)
vl_3d = my.imgs_hists_3d(vlImgs, bins = 8)
tr_3d_shape = []
for t in tr_3d:
    tr_3d_shape.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
vl_3d_shape = []
for t in vl_3d:
    vl_3d_shape.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
tr_descriptors = np.array(tr_3d_shape)
vl_descriptors = np.array(vl_3d_shape)  

tr_descriptors= tr_descriptors.astype('float32')
vl_descriptors= vl_descriptors.astype('float32')
    
    
clf = KNeighborsClassifier(n_neighbors= 25)
clf.fit(tr_descriptors, trLabels)
predictions = clf.predict(vl_descriptors)

Hitrate = []
Hitrate.append(sum(predictions == vlLabels))
acc = []
acc.append(sum(predictions == vlLabels)/len(vlLabels))
print(acc,Hitrate)
#
### Loop erstellen
#descrip = ['mean','sd','hist','3d','mean_std','mean_std_hist', 'mean_std_3d']
#neighbors = [20,25,30,35,50]
#bins = [4,8,10,12,14,16,18,20,22]

## Diskriptoren erstellen -----------------------------------------------------------------------------------------
#for i in descrip:
#    for j in neighbors:
#        
#        tr_means = np.array([])
#        vl_means = np.array([])
#        tr_std = np.array([])
#        vl_std = np.array([])
#        tr_hist = np.array([])
#        vl_hist = np.array([])
#        tr_3d_shape = np.array([])
#        vl_3d_shape = np.array([])
#        
##        # mean
#        if i == "mean":
#            tr_means = my.imgs_means(trImgs)
#            vl_means = my.imgs_means(vlImgs)
#            tr_descriptors = tr_means
#            vl_descriptors = vl_means
##        
#        ## sd
#        if i == "sd":
#            tr_std = my.imgs_std(trImgs)
#            vl_std = my.imgs_std(vlImgs)
#            tr_descriptors = tr_std
#            vl_descriptors = vl_std
#            
#        # hists
#        if i == "hist":
#            tr_hist = my.imgs_hists(trImgs, bins = 8)
#            vl_hist = my.imgs_hists(vlImgs, bins = 8)
#            tr_descriptors = np.array(tr_hist)
#            vl_descriptors = np.array(vl_hist)
#        
#        #3d
#        if i == "3d":
#            tr_3d = my.imgs_hists_3d(trImgs, bins = 16)
#            vl_3d = my.imgs_hists_3d(vlImgs, bins = 16)
#            tr_3d_shape = []
#            for t in tr_3d:
#                tr_3d_shape.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
#            vl_3d_shape = []
#            for t in vl_3d:
#                vl_3d_shape.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
#            tr_descriptors = np.array(tr_3d_shape)
#            vl_descriptors = np.array(vl_3d_shape)  
#        
#        if i == "mean_std":
#            tr_means = my.imgs_means(trImgs)
#            vl_means = my.imgs_means(vlImgs)
#            tr_std = my.imgs_std(trImgs)
#            vl_std = my.imgs_std(vlImgs)
#            tr_descriptors = np.concatenate((tr_means,tr_std), axis=1)
#            vl_descriptors = np.concatenate((vl_means,vl_std), axis=1)
#            
#        if "mean_std_hist" == i:
#            tr_means = my.imgs_means(trImgs)
#            vl_means = my.imgs_means(vlImgs)
#            tr_std = my.imgs_std(trImgs)
#            vl_std = my.imgs_std(vlImgs)
#            tr_hist = my.imgs_hists(trImgs, bins = 8)
#            vl_hist = my.imgs_hists(vlImgs, bins = 8)
#            tr_descriptors = np.concatenate((tr_means,tr_std,tr_hist), axis=1)
#            vl_descriptors = np.concatenate((vl_means,vl_std,vl_hist), axis=1)
#        
#        if "mean_std_3d" == i:
#            tr_means = my.imgs_means(trImgs)
#            vl_means = my.imgs_means(vlImgs)
#            tr_std = my.imgs_std(trImgs)
#            vl_std = my.imgs_std(vlImgs)
#            tr_3d = my.imgs_hists_3d(trImgs, bins = 16)
#            vl_3d = my.imgs_hists_3d(vlImgs, bins = 16)
#            tr_3d_shape = []
#            for t in tr_3d:
#                tr_3d_shape.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
#            vl_3d_shape = []
#            for t in vl_3d:
#                vl_3d_shape.append(t.reshape((t.shape[0]* t.shape[1] * t.shape[2])))
#            tr_descriptors = np.concatenate((tr_means,tr_std,tr_3d_shape), axis=1)
#            vl_descriptors = np.concatenate((vl_means,vl_std,vl_3d_shape), axis=1)
#        
#        
#        ## Klassifikation erstellen --------------------------------------------------------------------------------------------
#        tr_descriptors= tr_descriptors.astype('float32')
#        vl_descriptors= vl_descriptors.astype('float32')
#
#        
#        clf = KNeighborsClassifier(n_neighbors= j)
#        clf.fit(tr_descriptors, trLabels)
#        predictions = clf.predict(vl_descriptors)
#        
#        Hitrate = []
#        Hitrate.append(sum(predictions == vlLabels))
#        acc = []
#        acc.append(sum(predictions == vlLabels)/len(vlLabels))
#        print(i,j,acc,Hitrate)
#









#trLabels -=1
#vlLabels -=1
#trLabels = np_utils.to_categorical(trLabels, 16)
#vlLabels = np_utils.to_categorical(vlLabels, 16)
#tr_descriptors = tr_descriptors/16384.0
#vl_descriptors = vl_descriptors/16384.0
#
#
### Neuronales Netz als Klassifikator -----------------------------------
#model = Sequential()
#model.add(Dense(128, activation='relu', name='fc1', input_shape=(512,)))
#model.add(Dense(128, activation='relu', name='fc3')) 
#model.add(Dense(16, activation='softmax')) 
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
#    
#model.fit(tr_descriptors, trLabels, batch_size= 8 , epochs = 50, verbose = 1)
#    
#print(model.evaluate(vl_descriptors, vlLabels , verbose=1))