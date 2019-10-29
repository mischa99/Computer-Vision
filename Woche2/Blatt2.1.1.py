#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:08:47 2019
Blatt 2.1, ufgabe 1
@author: mikhail
"""

import numpy as np
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

#Bild laden
img=imread("./catG.png")
plt.imshow(img, cmap="Greys_r")

#Wieviel Pixel hat das Bild?
print(img.shape[0]*img.shape[1])

#Maximum, Minimum, Mittelwert
img_max=0
img_min=0
img_avg=0

for x in range (img.shape[0]):
    for y in range (img.shape[1]):
        img_avg+=img[x,y]
        if img[x,y] > img_max:
            img_max = img[x,y]
            
        elif img[x,y] < img_min:
            img_max = img[x,y]

img_avg= (img_avg / (img.shape[0]*img.shape[1]))

print(img_max)
print(img_min)
print(img_avg)

