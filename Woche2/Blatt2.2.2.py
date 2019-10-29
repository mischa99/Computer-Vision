#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:06:45 2019
Blatt 2.2, Aufgabe 2
@author: mikhail
"""
import numpy as np
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

def histComp(hist1,hist2):
    diff=hist1[0]-hist2[0]
    s=0
    for x in diff:
        s+=pow(x,2)
    return np.sqrt(s)


img1=imread("./satBilder/agri3.png")
img2=imread("./satBilder/agri6.png")
img3=imread("./satBilder/urban1.png")

hist1= np.histogram(img1, bins=8,range= (0,256))
hist2= np.histogram(img2, bins=8,range= (0,256))
hist3= np.histogram(img3, bins=8,range= (0,256))
print(histComp(hist1,hist2))
print(histComp(hist1,hist3)) 
print(histComp(hist2,hist3))  
