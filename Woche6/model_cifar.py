#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:21:54 2019

@author: JanStrich
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def model_cifar(tr_descriptors, vl_descriptors, tr_labels, vl_labels,
                layer = 3, neuron = 128, batch = 32,  epoche = 10):
    model = Sequential()
    
    if layer == 2:
        model.add(Dense(neuron, activation='relu', name='fc1', input_shape=(tr_descriptors.shape[1],)))
        model.add(Dense(neuron, activation='relu', name='fc2')) 
    if layer == 3:
        model.add(Dense(neuron, activation='relu', name='fc1', input_shape=(tr_descriptors.shape[1],)))
        model.add(Dense(neuron, activation='relu', name='fc2'))
        model.add(Dense(neuron, activation='relu', name='fc3')) 
    if layer == 4:
        model.add(Dense(neuron, activation='relu', name='fc1', input_shape=(tr_descriptors.shape[1],)))
        model.add(Dense(neuron, activation='relu', name='fc2'))
        model.add(Dense(neuron, activation='relu', name='fc3')) 
        model.add(Dense(neuron, activation='relu', name='fc4')) 
    if layer == 5:
        model.add(Dense(neuron, activation='relu', name='fc1', input_shape=(tr_descriptors.shape[1],)))
        model.add(Dense(neuron, activation='relu', name='fc2'))
        model.add(Dense(neuron, activation='relu', name='fc3')) 
        model.add(Dense(neuron, activation='relu', name='fc4')) 
        model.add(Dense(neuron, activation='relu', name='fc5'))
    
    model.add(Dense(10, activation='softmax')) 
    model.compile(loss='categorical_crossentropy',
    optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
    
    model.fit(tr_descriptors, tr_labels, batch_size= batch , epochs = epoche, verbose = 1)
    
    return model.evaluate(vl_descriptors, vl_labels , verbose=1)
    