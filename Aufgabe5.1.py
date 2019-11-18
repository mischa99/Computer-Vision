# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:22:28 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
import glob
import Funktionsscript as myi
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve


## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche5')


## 5.1.1
listfiles = glob.glob('*.png')
img = imread(listfiles[2])


## 5.1.2
img_new = img - img
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        pixel = img[x,y]
        if (x-1 >= 0 and x-1 < img.shape[0] and y-1 >= 0 and y-1 < img.shape[1]):
            top_left = img[x-1,y-1]
        else:
            top_left = 0
        if (x >= 0 and x < img.shape[0] and y-1 >= 0 and y-1 < img.shape[1]):
            middle_left = img[x,y-1]
        else:
            middle_left = 0
        if (x+1 >= 0 and x+1 < img.shape[0] and y-1 >= 0 and y-1 < img.shape[1]):
            buttom_left = img[x+1,y-1]
        else:
            buttom_left = 0
        if (x-1 >= 0 and x-1 < img.shape[0] and y >= 0 and y < img.shape[1]):
            top = img[x-1,y]
        else:
            top = 0
        if (x+1 >= 0 and x+1 < img.shape[0] and y >= 0 and y < img.shape[1]):
            buttom = img[x+1,y]
        else:
            buttom = 0
        if (x-1 >= 0 and x-1 < img.shape[0] and y+1 >= 0 and y+1 < img.shape[1]):
            top_right = img[x-1,y+1]
        else:
            top_right = 0
        if (x >= 0 and x < img.shape[0] and y+1 >= 0 and y+1 < img.shape[1]):
            middle_right = img[x,y+1]
        else:
            middle_right = 0
        if (x+1 >= 0 and x+1 < img.shape[0] and y+1 >= 0 and y+1 < img.shape[1]):
            buttom_right = img[x+1,y+1]
        else:
            buttom_right = 0
        
        temp = [pixel,top_left,middle_left,buttom_left, top,buttom,top_right,middle_right,buttom_right]
        img_new[x,y] = float(np.mean(temp))      
        
## Aufgabe 5.1.2
matrix = [[1/9,1/9,1/9],
          [1/9,1/9,1/9],
          [1/9,1/9,1/9]
          ]
import time
tic = time.time()
img_new2 = convolve(img,matrix)
toc = time.time()
diff = toc-tic
print(diff)



        