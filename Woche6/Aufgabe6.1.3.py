#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:02:13 2019

@author: JanStrich
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import numpy as np
import Funktionsscript as my
from sklearn.neighbors import KNeighborsClassifier
from model_cifar import model_cifar

## Daten einlesen
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


## Kennwerte bestimmen
tr_means = my.imgs_means(X_train)
vl_means = my.imgs_means(X_test)
tr_std = my.imgs_std(X_train)
vl_std = my.imgs_std(X_test)



## Descriptoren erstellen 
tr_descriptors = np.concatenate((tr_means,tr_std), axis=1)
vl_descriptors = np.concatenate((vl_means,vl_std), axis=1)
tr_descriptors= tr_descriptors.astype('float32')
vl_descriptors= vl_descriptors.astype('float32')

## Labels ins richtige Format bringen
y_train = y_train.ravel()
y_test = y_test.ravel()
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


## nearest nabour
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(tr_descriptors, y_train)
predictions = clf.predict(vl_descriptors)


Hitrate = 0
for i in range(len(y_test)):
    if predictions[i] == y_test[i]:
        Hitrate+= 1
acc = Hitrate/len(y_test)


## Neuronale Netze ###
## Kennwerte bestimmen
tr_means = my.imgs_means(X_train)/255
vl_means = my.imgs_means(X_test)/255
tr_std = my.imgs_std(X_train)/128
vl_std = my.imgs_std(X_test)/128

## Descriptoren erstellen 
tr_descriptors = np.concatenate((tr_means,tr_std), axis=1)
vl_descriptors = np.concatenate((vl_means,vl_std), axis=1)
tr_descriptors = tr_descriptors.astype('float32')
vl_descriptors = vl_descriptors.astype('float32')

#numImages = X_test.shape[0]
#vl_descriptors = np.array([X_test[i].flatten() for i in range(0,numImages)])/255
#numImages = X_train.shape[0]
#tr_descriptors = np.array([X_train[i].flatten() for i in range(0,numImages)])/255

#layer = [2,3,4,5]
#neuron = [2,4,8,16,32]
#batch = 32
#
#acc = []
#loss = []
#for i in layer:
#    for j in neuron:
#        hitrate = model_cifar(tr_descriptors, vl_descriptors, Y_train, Y_test, 
#                    layer = i, neuron = j, batch = 32, epoche = 100)
#        loss.append(hitrate[0])
#        acc.append(hitrate[1])
#        
#numImages = X_test.shape[0]
#vl_descriptors = np.array([X_test[i].flatten() for i in range(0,numImages)])/255
#numImages = X_train.shape[0]
#tr_descriptors = np.array([X_train[i].flatten() for i in range(0,numImages)])/255
#
#epoche = [10,50,100]
#loss2 = []
#acc2 = []
#for i in layer:
#    for j in neuron:
#        for k in epoche:
#            hitrate = model_cifar(tr_descriptors, vl_descriptors, Y_train, Y_test, 
#                        layer = i, neuron = j, batch = 32, epoche = k)
#            loss2.append(hitrate[0])
#            acc2.append(hitrate[1])