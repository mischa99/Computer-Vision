#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:01:55 2019
Blatt 2.1, Aufgabe 3
@author: mikhail
"""
import numpy as np
from skimage.io import imread,imsave

#Bild laden
img=imread("./catG.png")

img_min=np.min(img)
img_max=np.max(img)
img_avg=np.average(img)

print(img_min)
print(img_max)
print(img_avg)
print(np.size(img))
