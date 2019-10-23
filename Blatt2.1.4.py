#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:05:18 2019
Blatt 2.1, Aufgabe 4
@author: mikhail
"""
import numpy as np
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
#Bild laden
img=imread("./catG.png")
#4.1: Invetiert das Bild - hell->dunkel , dunkel ->hell
img_invert = np.invert(img)
plt.imshow(img_invert, cmap="Greys_r")
#4.2: spiegeln an der vertikalen Achse
img_flip = np.fliplr(img)
plt.imshow(img_flip, cmap="Greys_r")
#4.3 Bildausschnitt
plt.imshow(img[20:280,250:500],cmap="Greys_r")