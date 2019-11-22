#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:49:52 2019

@author: mikhail
"""
import bilderGenerator as bG
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import sobel_h, sobel_v, threshold_otsu, gaussian
#import Funktionsscript as my
from skimage.measure import regionprops
import matplotlib.pyplot as plt

#1.1
tr_mws, tr_stds, tr_labels = bG.zieheBilder(250)
vl_mws, vl_stds, vl_labels = bG.zieheBilder(25)

#1.2
'''
plt.close('all')
fig, ax = plt.subplots(1,1)
index_k1 = np.where(tr_labels==1)
tr_mws_k1=tr_mws[index_k1]
tr_stds_k1=tr_stds[index_k1]

ax.plot(tr_mws_k1,tr_stds_k1,'rx')

index_k2 = np.where(tr_labels==-1)
tr_mws_k2=tr_mws[index_k2]
tr_stds_k2=tr_stds[index_k2]

ax.plot(tr_mws_k2,tr_stds_k2,'rx')
'''

#1.3
w1=0.0001
w2=-0.0002
b=0.001
tr_y = []
for t in list(range(0,500,1)):
    tr_y.append(w1 *tr_mws[t] + w2*tr_stds[t] + b)
    


w1=0.0001
w2=-0.0002
b=0.001
vl_y = []
for t in list(range(0,50,1)):
    y=w1 *vl_mws[t] + w2*vl_stds[t] + b
    if y>=0:
        y=1
    else:
        y=-1
    vl_y.append(y)
    
hit=0
for t in list(range(0,50,1)):
    if(vl_y[t]==vl_labels[t]):
        hit+=1
        
print(hit/len(vl_labels))

#1.4

#initialisierung
w1=np.random.normal(0,0.001)
w2=np.random.normal(0,0.001)
b=0
hitrates=[]

for t in list(range(0,100,1)):
    tr_y = []
    for t in list(range(0,500,1)):
        y=w1 *tr_mws[t] + w2*tr_stds[t] + b  
        if(np.sign(y)==np.sign(tr_labels[t])):
            continue
        else:
            #partielle Ableitungen bestimmen, Formel siehe Folie
            d_w1=2*(w1 * tr_mws[t] + w2 * tr_stds[t] + b - tr_labels[t]) * tr_mws[t]
            d_w2=2*(w1 * tr_mws[t] + w2 * tr_stds[t] + b - tr_labels[t]) * tr_stds[t]
            d_b=2*(w1 * tr_mws[t] + w2 * tr_stds[t] + b - tr_labels[t])
            
            #Gewichte aktualisiern
            learning_rate=0.0000005
            w1=w1-learning_rate*d_w1
            w2=w2-learning_rate*d_w2
            b=b-learning_rate*d_b
               
    vl_y = []
    for t in list(range(0,50,1)):
        y=w1 *vl_mws[t] + w2*vl_stds[t] + b
        if y>=0:
            y=1
        else:
            y=-1
        vl_y.append(y)
        
    hit=0
    for t in list(range(0,50,1)):
        if(vl_y[t]==vl_labels[t]):
            hit+=1
            
    hitrates.append(hit/len(vl_labels))      
    
print(hitrates)

#1.7
line_x=[]
line_y=[]
for x in list(range(0,255,1)):
    for y in list(range(0,128,1)):
        y_end=w1 *x + w2*y + b 
        if y_end<0.0001:
            line_x.append(x)
            line_y.append(y)


plt.close('all')
fig, ax = plt.subplots(1,1)
index_k1 = np.where(tr_labels==1)
tr_mws_k1=tr_mws[index_k1]
tr_stds_k1=tr_stds[index_k1]

ax.plot(tr_mws_k1,tr_stds_k1,'rx')

index_k2 = np.where(tr_labels==-1)
tr_mws_k2=tr_mws[index_k2]
tr_stds_k2=tr_stds[index_k2]
ax.plot(tr_mws_k2,tr_stds_k2,'bx')

ax.plot(line_x,line_y,'gx')           
