#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:00:30 2019

@author: 7dill
"""

from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import Funktionsscript as my
import glob

#always do this before working with keras!
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

import tensorflow as tf
tf.random.set_seed(123)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator




'''
#Daten laden
tr_listfiles = glob.glob('GroceryStoreDataset-V5/train/*.jpg')
X_train = []
trLabels = []
for t in range(len(tr_listfiles)):
    X_train.append(imread(tr_listfiles[t]))
    trLabels.append(glob.glob('GroceryStoreDataset-V5/train/*.jpg')[t].split('/')[-1].split('.')[0])
    
vl_listfiles = glob.glob('GroceryStoreDataset-V5/val/*.jpg')
X_test = []
vlLabels = []
for t in range(len(vl_listfiles)):
    X_test.append(imread(vl_listfiles[t]))
    vlLabels.append(glob.glob('GroceryStoreDataset-V5/val/*.jpg')[t].split('/')[-1].split('.')[0])
        
# Labels umbenennen
Y_train = my.imgs_labels(trLabels)
Y_test = my.imgs_labels(vlLabels)


'''

train_datagen = ImageDataGenerator(
        rescale=1./255
        )

val_datagen = ImageDataGenerator(
        rescale=1./255
        )

train_generator = train_datagen.flow_from_directory(
    directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/train/",
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/val/",
        target_size=(256, 256),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
        seed=42
        )

'''
#Datentyp anpassen
for img in X_train:
    img = img.astype('float32')

for img in X_test:
    img = img.astype('float32')
#Wertebereich normieren, auf zwischen 0 und 1
for img in X_train:
    img=img/255

for img in X_test:
    img = img/255

# Label  umformatieren 
Y_train = np_utils.to_categorical(Y_train, 16)
Y_test = np_utils.to_categorical(Y_test, 16)
'''

#CNN bauen
model = Sequential()
#input conv layer 
model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', padding='same', input_shape=(256, 256, 3)))
#hidden conv layer
model.add(Conv2D(32, (3, 3), activation='relu', name='conv2', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

model.add(Conv2D(64, (3, 3), activation='relu', name='conv3', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv4', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

model.add(Flatten())
#hidden layer
model.add(Dense(256, activation='relu', name='fc1')) 
#output layer,
model.add(Dense(16, activation='softmax')) 

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

#CNN trainieren
model.fit_generator(train_generator,steps_per_epoch=100, epochs=10, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='valloss',min_delta=0, patience=3),
        ModelCheckpoint(filepath='/informatik2/students/home/7dill/Desktop/CV/Project/cnn1weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

model.load_weights('cnn1weights.h5', by_name=True)

val_loss, val_acc = model.evaluate(X_test, Y_test, verbose=1)
print(val_loss, val_acc)