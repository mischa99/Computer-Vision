#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:32:42 2019
Blatt 2.1 Aufgabe 2
@author: mikhail
"""
import numpy as np
from skimage.io import imread,imsave
#Bild laden
img=imread("./catG.png")

#Wieviele weiße und scharze Pixel?
white_pixels = np.argwhere(img==0)
black_pixels = np.argwhere(img==255)
print(len(white_pixels))
print(len(black_pixels))

#Wie viele Graustufen im Bild?

#np.unique returns sorted unique elements of an array
#return_counts=True -> returns number of times each value comes up
unique, counts = np.unique(img,return_counts=True)
print(dict(zip(unique,counts)))

#print(np.bincount(img))

#
