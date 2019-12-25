#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:04:46 2019

@author: 7dill
"""


from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D, BatchNormalization
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

train_generator = train_datagen.flow_from_directory(
    directory="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/train/",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True, 
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/val/",
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
#flat1 = Flatten()(model.output)
fc1 = Dense(128, activation='relu')(pool1)
batch_norm1=BatchNormalization(axis=1)(fc1)
fc2 = Dense(128, activation='relu')(batch_norm1)
batch_norm2=BatchNormalization(axis=1)(fc2)
predictions = Dense (16, activation= 'softmax')(batch_norm2)
model = Model(inputs = model.input, outputs = predictions)
#print(model.summary())

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=2.1076967e-06, momentum=0.9), metrics=['accuracy'])

model.load_weights('InceptionV3weights.h5', by_name=True)

#CNN trainieren
model.fit_generator(train_generator, epochs=10, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=3),
        ModelCheckpoint(filepath='/informatik1/students/home/9raudin/Desktop/CV/Project/InceptionV3bweights.h5',
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)])

model.load_weights('InceptionV3bweights.h5', by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)


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

