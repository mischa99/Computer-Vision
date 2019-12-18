#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:04:46 2019

@author: 7dill
"""


from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications


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
    directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/train/",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/val/",
        target_size=(299, 299),
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )
'''
test_generator = test_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/test/",
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )
'''
new_input = Input(shape=(299,299,3))
model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=new_input,pooling='avg')
#print(model.summary())
#plot_model(model,to_file='vgg.png')

#predictions = model.predict_generator(test_generator,steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
#print(predictions)

# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False

# add new classifier layers
#x = GlobalAveragePooling2D()(model.output)
flat1 = Flatten()(model.output)
fc1 = Dense(256, activation='relu')(flat1)
fc2 = Dense(256, activation='relu')(fc1)
output = Dense(16, activation='softmax')(fc2)
# define new model
model = Model(inputs=model.inputs, outputs=output)

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

#CNN trainieren
model.fit_generator(train_generator, epochs=10, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=3),
        ModelCheckpoint(filepath='/informatik2/students/home/7dill/Desktop/CV/Project/InceptionV3weights.h5', 
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

model.load_weights('InceptionV3weights.h5', by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)
