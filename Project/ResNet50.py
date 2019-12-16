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