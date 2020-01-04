#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:14:49 2019

@author: 9raudin
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D, BatchNormalization
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V7_CNN"
FILEPATH_WEIGHTS= "/informatik1/students/home/9raudin/Desktop/CV/Project/"
weights_name="-1"



new_input = Input(shape=(299,299,3)) #define Input Layer Shape for Inception Model
model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=new_input)

#mark all loaded Inception layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers
#x = GlobalAveragePooling2D()(model.output)
flat1 = Flatten()(model.output)
fc1 = Dense(256, activation='relu')(flat1)
#drop1 = Dropout(0.2)(fc1)
#batch_norm1=BatchNormalization(axis=1)(drop1)
fc2 = Dense(128, activation='relu')(fc1)
#drop2 = Dropout(0.2)(fc2)
#batch_norm2=BatchNormalization(axis=1)(drop2)
output = Dense (16, activation= 'softmax')(fc2)

# define new model
model = Model(inputs=model.inputs, outputs=output)

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255
        )

val_datagen = ImageDataGenerator(
        rescale=1./255
        )


train_generator = train_datagen.flow_from_directory(
    directory=FILEPATH_DATA + "/train/",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=4,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "/train/",
        target_size=(299, 299),
        batch_size=4,
        class_mode='categorical',
        shuffle=True,
        )
   

#CNN trainieren
history = model.fit_generator(train_generator, epochs=200, validation_data=val_generator,
          verbose=1)# callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=4),
        #ModelCheckpoint(filepath=FILEPATH_WEIGHTS+weights_name,
        #monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

#model.load_weights(weights_name, by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)


#create pdf to store loss and accuracy graphs
pdf = PdfPages('/informatik1/students/home/9raudin/Desktop/CV/Project/loss_acc_graph.pdf')
#plot model loss
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
#plt.show() #show in window/terminal, doesn't work over ssh

pdf.savefig(fig1) #add loss figure to pdf
plt.close()

# plot model accuracy
fig2 = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()

pdf.savefig(fig2) #add acc figure to pdf
plt.close()

pdf.close() #close pdf after adding two plots
