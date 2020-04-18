#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:14:49 2019

@author: Mikhail Raudin, Timm Dill
"""

from matplotlib import pyplot as plt

import random as rn
import numpy as np
import tensorflow as tf
import time

import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D, BatchNormalization
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

#setting seeds for numpy, tensorflow, random to create reproducable results
np.random.seed(123)
tf.random.set_seed(123)
rn.seed(123)

FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/DataX3_Final"
FILEPATH= "/informatik1/students/home/9raudin/Desktop/CV/Project/"

NAME="vgg16_256_03_256_02_Adam2-{}".format(int(time.time()))

text_file = open(FILEPATH + "Adam_lr_final2.txt", "a")

max_count=100 #try 100 random learning rates
for count in range(max_count):
  lr = 10**rn.uniform(-3,-5) #random search between 10^-3 and 10^-6
  print(count,"/",max_count) #show iteration number

  new_input=Input(shape=(224,224,3))
  model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

  #mark loaded layers as not trainable
  for layer in model.layers:
      layer.trainable = False

  # add new classifier layers
  flat1 = Flatten()(model.output)
  fc1 = Dense(256, activation='relu')(flat1)
  drop1 = Dropout(0.3)(fc1)
  fc2 = Dense(256, activation='relu')(drop1)
  drop2 = Dropout(0.2)(fc2)
  output = Dense (16, activation= 'softmax')(drop2)

  # define new model
  model = Model(inputs=model.inputs, outputs=output)

    
  train_datagen = ImageDataGenerator(
        rescale=1./255
        )

  val_datagen = ImageDataGenerator(
        rescale=1./255
        )


  train_generator = train_datagen.flow_from_directory(
    directory=FILEPATH_DATA + "/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    
  )

  val_generator = val_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "/val/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )

  #CNN komplieren
  model.compile(loss='categorical_crossentropy',
  optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])

  tensorboard = TensorBoard(log_dir="logs/{}".format(NAME + "_" + str(lr)))
  #train CNN only for 5 epochs, enough to see if lr works or not
  model.fit_generator(train_generator, epochs=5, validation_data=val_generator,
          verbose=0, callbacks=[tensorboard])

  val_loss, val_acc = model.evaluate_generator(val_generator,
  callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=0)
  
  #write each loops result in a text file
  loop_nr = [str(count), "/", str(max_count) + "\n"]
  results = ["val_acc:", str(val_acc),"lr:", str(lr) + "\n"]
  text_file.writelines(loop_nr)
  text_file.writelines(results)

  print("val_acc:",val_acc,"lr:",lr, count,"/",max_count) #print learning rate and val_accuracy in terminal
    
 
text_file.close()


