# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:30:57 2019

@author: Jan
"""

import os
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import sobel_h, sobel_v, threshold_otsu, gaussian
from skimage.feature import match_template
import glob
import Funktionsscript as myi
from skimage.measure import regionprops
import matplotlib.pyplot as plt

## Change Direction
os.chdir('C:\\Users\\Jan\Dropbox\\Uni\\Informatik\\Computer Vision\\Computer-Vision\\Woche5')


## Katzenbild einladen
listfiles = glob.glob('*.png')
cat = imread(listfiles[1])
cateye = imread(listfiles[0])
wally = imread(listfiles[4])
listfiles2 = glob.glob('*.jpg')
wally_img = imread(listfiles2[0])

result = match_template(cat, cateye, pad_input=True)
plt.imshow(result)

print((np.argmax(result), result.shape))



### Wally 
result = match_template(wally_img, wally, pad_input=True)
plt.imshow(result)

print((np.argmax(result), result.shape))
