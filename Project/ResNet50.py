from skimage.io import imread, imsave
import matplotlib.pyplot as plt

#always do this before working with keras!
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(123)
#tf.random.set_seed(123)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications

import os
from skimage.io import imread

#Lambda-Layer vor dem ersten CNN Layer für resize images 
'''
original_dim = (32, 32, 3)
target_size = (64, 64)
input = keras.layers.Input(original_dim)
x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, target_size))(input)
'''

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
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/val/",
        target_size=(256, 256),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
        seed=42
        )

#include_top = false -> not including output layers because we need to fit the model on our own problem 
model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))
x = model.output
x = GlobalAveragePooling2D()(x)
#x = Dropout(0.2)(x)
predictions = Dense (16, activation= 'softmax')(x)
model = Model(inputs = model.input, outputs = predictions)

model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=100, epochs=10, validation_data=val_generator,
          verbose=1, callbacks=[EarlyStopping(monitor='valloss',min_delta=0, patience=3),
        ModelCheckpoint(filepath='/informatik2/students/home/7dill/Desktop/CV/Project/cnn1weights.h5',
         monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])


test_datagen = ImageDataGenerator(
        rescale=1./255
        )

test_generator = train_datagen.flow_from_directory(
    directory="/informatik2/students/home/7dill/Desktop/CV/Project/GroceryStoreDataset-V6_CNN/test/",
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
)

val_loss, val_acc = model.evaluate_generator(test_generators,steps=100,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)

print(val_loss, val_acc)










'''
DATADIR = 
CATEGORIES = ["Apple", "Aubergine", "Banana", "Carrots", "Cucumber", "Ginger", "Lemon", "Melon", "Orange", "Pear",
            "Pepper","Pineapple", "Potato", "Tomato", "Watermelon", "Zucchini"]

for category in CATEGORIES:
    path= os.path.join(DATADIR, category) # path to Image Class dir 
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break
'''


