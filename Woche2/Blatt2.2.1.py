#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:54:38 2019
Blatt 2.2, Aufgabe 1
@author: mikhail
"""
import numpy as np
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

def picCompare(img1, img2):
    diff = np.mean(img1)-np.mean(img2)
    return abs(diff)

img1=imread("./satBilder/agri3.png")
img2=imread("./satBilder/agri6.png")
img3=imread("./satBilder/urban1.png")
print(picCompare(img1,img2))
print(picCompare(img1,img3))
print(picCompare(img2,img3))

