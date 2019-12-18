#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 07:59:10 2019

@author: JanStrich
"""

## Neuronales Netz als Klassifikator -----------------------------------
model = Sequential()
model.add(Dense(128, activation='relu', name='fc1', input_shape=(512,)))
model.add(Dense(128, activation='relu', name='fc2'))
model.add(Dense(128, activation='relu', name='fc3')) 
model.add(Dense(16, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
    
model.fit(tr_descriptors, trLabels, batch_size= 32 , epochs = 250, verbose = 1)
    
print(model.evaluate(vl_descriptors, vlLabels , verbose=1))