#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:16:28 2019

@author: mikhail
"""
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

#always do this before working with keras!
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(123)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Datentyp anpassen
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#Wertebereich normieren, auf zwischen 0 und 1
X_train /= 255
X_test /= 255

# Label  umformatieren 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
''' 
Aufgabe 1.11
#Data Augmentation: we flip the train img vertically to create more train data for the CNN
X_train_flipped = np.flipud(X_train)
X_more_train = np.concatenate(X_train,X_train_flipped,axis=0)
#adjust labels
Y_train=np.concatenate(Y_train,Y_train,axis=0)
'''

#CNN bauen
model = Sequential()
#input conv layer 
model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', padding='same', input_shape=(32, 32, 3)))
#hidden conv layer
# Aufg 1.6 model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', padding='same'))
#Aufg 1.7 model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
'''
Aufagbe 1.8
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv1', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
'''
model.add(Flatten())
#hidden layer
model.add(Dense(256, activation='relu', name='fc1')) 
#output layer,
model.add(Dense(10, activation='softmax')) 

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

#CNN trainieren
model.fit(X_train, Y_train, batch_size=32, epochs=20,
          validation_split= 0.2, verbose=1, callbacks=EarlyStopping(monitor='valloss',mindelta=0, patience=3)
        #Aufgabe 1.5  ModelCheckpoint(filepath='/Users/mikhail/Desktop/Praktikum_CV/Uebungen_CV/cifar10weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1))

#model.load_weights('cifar10weights.h5', by_name=True)

score = model.evaluate(X_test, Y_test, verbose=1)
print(score)

