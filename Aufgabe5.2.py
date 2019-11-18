# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:22:28 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import sobel_h, sobel_v, threshold_otsu, gaussian
import glob
import Funktionsscript as myi
from skimage.measure import regionprops
import matplotlib.pyplot as plt

## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche5')


## Katzenbild einladen
listfiles = glob.glob('*.png')
img = imread(listfiles[1])
img2 = imread(listfiles[2])


##Aufgabe 5.2.1
img_h = sobel_h(img)
img_v = sobel_v(img)
plt.imshow(img_h, cmap = "Greys_r")
plt.imshow(img_v, cmap = "Greys_r")

##Aufgabe 5.2.2
img2_h = sobel_h(img2)
img2_v = sobel_v(img2)
sobel=np.hypot(img2_h,img2_v)
plt.imshow(sobel, cmap = "Greys_r")

img2 = gaussian(img2, 5)
img2_h = sobel_h(img2)
img2_v = sobel_v(img2)
sobel=np.hypot(img2_h,img2_v)
plt.imshow(sobel, cmap = "Greys_r")
