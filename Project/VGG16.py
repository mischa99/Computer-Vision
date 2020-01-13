#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:00:30 2019
@author: 7dill
"""
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import numpy as np
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
from sklearn.metrics import confusion_matrix, classification_report
import os
from skimage.io import imread
import random
import pickle
import pandas as pd
from sklearn.utils import resample


train_datagen = ImageDataGenerator(
        rescale=1./255
        )

val_datagen = ImageDataGenerator(
        rescale=1./255
        )

test_datagen = ImageDataGenerator(
        rescale=1./255
        )

train_generator = train_datagen.flow_from_directory(
    directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V9_CNN/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V9_CNN/val/",
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )

test_generator = test_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V9_CNN/test/",
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=False,
        )

new_input = Input(shape=(224,224,3))
model = applications.vgg16.VGG16(weights='imagenet', input_tensor=new_input, include_top=False)
#print(model.summary())
#plot_model(model,to_file='vgg.png')

#predictions = model.predict_generator(test_generator,steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
#print(predictions)

# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False

# add new classifier layers
pool1= GlobalAveragePooling2D()(model.output)
#flat1 = Flatten()(model.output)
fc1 = Dense(256, activation='relu')(pool1)
#drop1 = Dropout(0.2)(fc1)
fc2 = Dense(256, activation='relu')(fc1)
#fc3 = Dense(256, activation='relu')(fc2)
output = Dense(16, activation='softmax')(fc2)
# define new model
model = Model(inputs=model.inputs, outputs=output)

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

#CNN trainieren
model.fit_generator(train_generator, epochs=20, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=20),
        ModelCheckpoint(filepath='/informatik2/students/home/7dill/Desktop/CV/Project/VGG16weights.h5', 
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

model.load_weights('VGG16weights.h5', by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)




model.load_weights("VGG16weights.h5", by_name=True)

predictions = model.predict_generator(test_generator, verbose=1)

predicted_classes=np.argmax(predictions,axis=1)
class_labels = list(test_generator.class_indices.keys())   

cm = confusion_matrix(test_generator.classes, np.round(predictions[:,0]))
print(cm)
#disp = plot_confusion_matrix =(cm,cmap=plt.cm.Blues)

pdf = PdfPages('/informatik2/students/home/7dill/Desktop/CV/Project/vgg16_1_cm.pdf')
fig1=plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title('Confusion matrix ')
plt.colorbar()
plt.close()
pdf.savefig(fig1) #add cm to pdf
pdf.close()

report = classification_report(test_generator.classes, predicted_classes, target_names=class_labels)
print(report)   
