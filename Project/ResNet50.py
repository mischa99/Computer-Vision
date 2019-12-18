#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:02:32 2019

@author: 7dill
"""


from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D, Input
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255
        )

val_datagen = ImageDataGenerator(
        rescale=1./255
        )

train_generator = train_datagen.flow_from_directory(
    directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/val/",
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
        seed=42
        )

#include_top = false -> not including output layers because we need to fit the model on our own problem 
new_input = Input(shape=(224,224,3))
model = applications.resnet50.ResNet50(weights='imagenet', include_top=False,input_tensor=new_input)

# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False
    
# adding our own classifier/output layers
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
fc1 = Dense(256, activation='relu')(x)
fc2 = Dense(256, activation='relu')(fc1)
predictions = Dense (16, activation= 'softmax')(fc2)
model = Model(inputs = model.input, outputs = predictions)

model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

model.fit_generator(train_generator, epochs=10, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=3),
        ModelCheckpoint(filepath='/informatik2/students/home/7dill/Desktop/CV/Project/ResNet50weights.h5',
         monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

model.load_weights('ResNet50weights.h5', by_name=True)
val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)

print(val_loss, val_acc)
