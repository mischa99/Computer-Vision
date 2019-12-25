#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:14:49 2019

@author: 7dill
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:00:30 2019

@author: 7dill
"""
from matplotlib import pyplot as plt
import random as rn
import numpy as np
import tensorflow as tf
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

#setting seeds for numpy, tensorflow, random to create reprducable results
np.random.seed(123)
tf.random.set_seed(123)
rn.seed(123)


FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V6_CNN"
FILEPATH_WEIGHTS= "/informatik1/students/home/9raudin/Desktop/CV/Project/"
weights_name="VGG16_2weights.h5"

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
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )
   

new_input=Input(shape=(224,224,3))
model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

#mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers
#x = GlobalAveragePooling2D()(model.output)
flat1 = Flatten()(model.output)
fc1 = Dense(128, activation='relu')(flat1)
batch_norm1=BatchNormalization(axis=1)(fc1)
fc2 = Dense(128, activation='relu')(batch_norm1)
batch_norm2=BatchNormalization(axis=1)(fc2)
output = Dense (16, activation= 'softmax')(batch_norm2)
# define new model
model = Model(inputs=model.inputs, outputs=output)


max_count=100
lr_array=[[],[]]
for count in range(max_count):
    lr = 10**rn.uniform(-3,-6)
    print(count,"/",max_count)
    #CNN komplieren
    model.compile(loss='categorical_crossentropy',
    optimizer=SGD(lr=lr, momentum=0.9), metrics=['accuracy'])


    #CNN trainieren
    history = model.fit_generator(train_generator, epochs=5, validation_data=val_generator,
          verbose=0)
          #callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=3)])
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