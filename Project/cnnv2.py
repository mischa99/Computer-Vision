#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:00:30 2019

@author: 7dill
"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


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
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/val/",
        target_size=(256, 256),
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )

#CNN bauen
model = Sequential()
#input conv layer 
model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', padding='same', input_shape=(256, 256, 3)))
#hidden conv layer
model.add(Conv2D(32, (3, 3), activation='relu', name='conv2', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, (3, 3), activation='relu', name='conv3', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', name='conv4', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(Dropout(rate=0.25))

model.add(Flatten())
#hidden layer
model.add(Dense(256, activation='relu', name='fc1'))
model.add(Dropout(rate=0.50)) 
#output layer,
model.add(Dense(16, activation='softmax')) 

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

#CNN trainieren
model.fit_generator(train_generator, steps_per_epoch=524, epochs=20, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=3),
        ModelCheckpoint(filepath='/informatik2/students/home/7dill/Desktop/CV/Project/cnn2weights.h5', 
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

model.load_weights('cnn2weights.h5', by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,steps_per_epoch=524,
callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)
