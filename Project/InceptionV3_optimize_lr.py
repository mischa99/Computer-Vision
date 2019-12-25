#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:04:46 2019

@author: 9raudin
"""

from matplotlib import pyplot as plt
import random
import numpy as np
import keras.backend as K
import os
from skimage.io import imread,imsave
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D, BatchNormalization
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications


FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V6_CNN"
FILEPATH_WEIGHTS= "/informatik1/students/home/9raudin/Desktop/CV/Project/"
weights_name="InceptionV3_4weights.h5"
'''
CATEGORIES = ["Apple", "Aubergine", "Banana", "Carrots", "Cucumber", "Ginger", "Lemon", "Melon", "Orange", "Pear", "Pepper", "Pineapple", "Potato", "Tomato", "Watermelon", "Zucchini"]

training_data=[]
for category in CATEGORIES:  # do dogs and cats

    path = os.path.join(FILEPATH_DATA + "/train/" ,category)  # create path to dogs and cats
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

    for img in os.listdir(path):  # iterate over each image per dogs and cats
        try:
            img_array = imread(os.path.join(path,img))  # convert to array
            np.append(training_data,(img_array))  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass
print(training_data)
'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        #featurewise_center = True, #create zero-centered input data
        #featurewise_std_normalization = True
        )

val_datagen = ImageDataGenerator(
        rescale=1./255,
        #featurewise_center=True,
        #featurewise_std_normalization=True
        )
 
#fit the data preprocessing
#train_datagen.fit(training_data)

#setup generator
train_generator = train_datagen.flow_from_directory(
    directory= FILEPATH_DATA + "/train/",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "/val/",
        target_size=(299, 299),
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )

new_input = Input(shape=(299,299,3))
model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=new_input)

for layer in model.layers:
    layer.trainable = False
    
pool1 = GlobalAveragePooling2D()(model.output)
#drop1 = Dropout(0.2)(pool1)
#flat1 = Flatten()(pool1)
fc1 = Dense(128, activation='relu')(pool1)
#drop1 = Dropout(0.2)(fc1)
batch_norm1=BatchNormalization(axis=1)(fc1)
fc2 = Dense(128, activation='relu')(batch_norm1)
#drop2 = Dropout(0.2)(fc2)
batch_norm2=BatchNormalization(axis=1)(fc2)
predictions = Dense (16, activation= 'softmax')(batch_norm2)
model = Model(inputs = model.input, outputs = predictions)


max_count=100
lr_array=[[],[]]
for count in range(max_count):
    lr = 10**random.uniform(-2,-3)
    print(count,"/",max_count)
    #CNN komplieren
    model.compile(loss='categorical_crossentropy',
    optimizer=SGD(lr=lr, momentum=0.9), metrics=['accuracy'])


    #CNN trainieren
    history = model.fit_generator(train_generator, epochs=5, validation_data=val_generator,
          verbose=0, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=3)])
        #ModelCheckpoint(filepath=FILEPATH_WEIGHTS + weights_name,
       # monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)])

    #model.load_weights(weights_name, by_name=True)

    val_loss, val_acc = model.evaluate_generator(val_generator,
    callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=0)
    print("val_acc:",val_acc,"lr:",K.eval(model.optimizer.lr), count,"/",max_count) #print learning rate
    lr_array[0].append(lr)
    lr_array[1].append(val_acc)

print(np.amax(lr_array,axis=0))
print(np.where(lr_array=np.amax(lr_array,axis=0)))
print(lr_array)
'''
#plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# plot model accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''
# fc1 128 -> drop 0.2 -> fc128 -> drop 0.2  .... lr =0.01 ... epochs20
# -> nach 13/20 mit 92,68% abgebrochen (loss von 0,00023
'''
V3: 256 -> 0.25 -> 256 -> 0.25 .... lr.=0.001 ... epochs 50
-> nach 7/50 mit 2.446119546890259 0.8956999182701111 abgebrochen
'''
