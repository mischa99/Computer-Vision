from skimage.io import imread, imsave
import matplotlib.pyplot as plt

#always do this before working with keras!
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

import tensorflow as tf
#from tensorflow import set_random_seed
#set_random_seed(123)
tf.random.set_seed(123)


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
