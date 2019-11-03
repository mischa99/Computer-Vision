# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:20:59 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche3')
img_color = imread("cat.png")

## Aufgabe 1.2
red = img_color[:,:,0]
green = img_color[:,:,1]
blue = img_color[:,:,2]

new_img = red + green + blue
plt.imshow(new_img, cmap = "Greys_r")



## Aufgabe 1.3
plt.imshow(red, cmap="Greys_r")
plt.imshow(green, cmap="Greys_r")
plt.imshow(blue, cmap="Greys_r")

## Aufgabe 1.4
img_wrong_color = np.dstack((green, red, blue))
plt.imshow(img_wrong_color)
# Grün = Red & Red = Grün

## Aufgabe 1.5
plt.imshow(255-img_color)

##Aufgabe 1.6
means = np.mean(img_color, axis = (0,1))
stds = np.std(img_color, axis = (0,1))
