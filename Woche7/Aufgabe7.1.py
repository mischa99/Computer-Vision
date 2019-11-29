#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:17:32 2019

@author: 7dill
"""
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

import tensorflow as tf
tf.random.set_seed(123)# -''-

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu",padding="same", name = "Conv", input_shape = (32,32,3)))
model.add(Flatten())
model.add(Dense(256, activation="relu", name="fc1"))
model.add(Dense(10, activation="softmax")) 

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size= 32, epochs = 20, verbose = 1, validation_split = 0.2, callbacks = [EarlyStopping(monitor='val_accuracy',min_delta=0.5, patience=2)])
rate = model.evaluate(X_test, Y_test , verbose=1)

