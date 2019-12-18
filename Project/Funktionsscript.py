# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:53:40 2019

@author: Jan
"""

import numpy as np
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import re

#### Operationen ------------------------------------------------------------------------
def euk_diff(zahl1, zahl2):
    return np.sum((zahl1-zahl2)**2)**.5

def intersec(hist1, hist2):
    sm = 0
    for i in range(len(hist1)):
        sm += min(hist1[i], hist2[i])
    return sm

def imgs_means(img_vector):
    vector = []
    for t in range(len(img_vector)):
        vector.append(np.mean(img_vector[t], axis = (0,1)))
    return np.array(vector)

def imgs_std(img_vector):
    vector = []
    for t in range(len(img_vector)):
        vector.append(np.std(img_vector[t], axis = (0,1)))
    return np.array(vector)

def imgs_ecc(img_vector):
    img_vector = imgs_to_grey(img_vector)
    otsu = []
    for t in range(len(img_vector)):
        otsu.append(threshold_otsu(img_vector[t]))
    mask = []
    for t in range(len(img_vector)):
        mask.append(img_vector[t] < otsu[t])
    img_vector = mask
    vector = []
    for t in range(len(img_vector)):
        vector.append(regionprops(img_vector[t].astype(np.int), coordinates= 'xy')[0].eccentricity)
    return vector
    
def imgs_hists(img_vector, bins = 8, withbins= False):
    vector = []
    if (withbins):
        for t in range(len(img_vector)):
            vector.append(np.histogram(img_vector[t], bins = bins, range = (0,256)))
    else:
        for t in range(len(img_vector)):
            vector.append(np.histogram(img_vector[t], bins = bins, range = (0,256))[0])
    return np.array(vector)


def imgs_hists_3d(img_vector, bins = 8, withbins= False):
    vector = []
    if (withbins):
        for t in range(len(img_vector)):
            img = img_vector[t].reshape((img_vector[t].shape[0]* img_vector[t].shape[1],3))
            vector.append(np.histogramdd(img, bins = [bins,bins,bins], range = ((0,256),(0,256),(0,256))))
    else:
        for t in range(len(img_vector)):
            img = img_vector[t].reshape((img_vector[t].shape[0]* img_vector[t].shape[1],3))
            vector.append(np.histogramdd(img, bins = [bins,bins,bins], range = ((0,256),(0,256),(0,256)))[0])
    return np.array(vector)

def imgs_to_grey(img_vector):
    vector = []
    for t in img_vector:
       vector.append(t[:,:,0]/3 + t[:,:,1]/3 + t[:,:,2]/3)
    return vector











### Funktionen zum berechen ---------------------------------------------------------------------

def imgs_nearest_neighbor(tr_vector, vl_vector, typ = "mean", verfahren = "euk"):
    best = []
    temp = []
    
    if (verfahren == "int" and (typ == "mean" or typ == "std")):
        return print("Intersection ist nur bei Histograms erlaubt")
    
    if (typ == "mean"):
        tr_vector = imgs_means(tr_vector)
        vl_vector = imgs_means(vl_vector)
    elif (typ == "hist"):
        tr_vector = imgs_hists(tr_vector)
        vl_vector = imgs_hists(vl_vector)
    elif (typ == "hist3d"):
        tr_vector = imgs_hists_3d(tr_vector)
        vl_vector = imgs_hists_3d(vl_vector)
    elif (typ == "std"):
        tr_vector = imgs_std(tr_vector)
        vl_vector = imgs_std(vl_vector)
    elif (typ == "ecc"):
        tr_vector = imgs_ecc(tr_vector)
        vl_vector = imgs_ecc(vl_vector)
    else:
        return print("Falscher Typ angegeben")
    
    if (verfahren == "euk"):
        for i in range(len(vl_vector)):
            for j in range(len(tr_vector)):
                temp.append(euk_diff(vl_vector[i],tr_vector[j]))
            best.append(np.argmin(temp))
            temp = []
    elif (verfahren == "int"):
        for i in range(len(vl_vector)):
            for j in range(len(tr_vector)):
                temp.append(intersec(vl_vector[i],tr_vector[j]))
            best.append(np.argmin(temp))
            temp = []
    return best

def imgs_hitrate(tr_labels, vl_labels, kate_test):
    kate = []     
    for t in kate_test:
        kate.append(tr_labels[t])
    
    i = 0
    for u in range(len(kate_test)):
        if (kate[u] == vl_labels[u]):
            i+= 1
    return  i/len(kate_test)

def imgs_bb(trImgs, vlImgs, Schwellenwert = "otsu"):
        
    trImgs_grey = imgs_to_grey(trImgs)
    vlImgs_grey = imgs_to_grey(vlImgs)
    
    if (Schwellenwert == "otsu"):
        tr_otsu = []
        vl_otsu = []
        for t in range(len(trImgs_grey)):
            tr_otsu.append(threshold_otsu(trImgs_grey[t]))
        for t in range(len(vlImgs_grey)):
            vl_otsu.append(threshold_otsu(vlImgs_grey[t]))
            
        trImgs_mask = []
        for t in range(len(trImgs_grey)):
            trImgs_mask.append(trImgs_grey[t] < tr_otsu[t])
        vlImgs_mask = []
        for t in range(len(vlImgs_grey)):
            vlImgs_mask.append(vlImgs_grey[t] < vl_otsu[t])
    else:
        trImgs_mask = []
        for t in trImgs_grey:
            trImgs_mask.append(t < Schwellenwert)
        vlImgs_mask = []
        for t in vlImgs_grey:
            vlImgs_mask.append(t < Schwellenwert)
        
            
    ## Int Casten für alle Elemente in der Maske
    for t in range(len(trImgs_mask)):
        trImgs_mask[t] = trImgs_mask[t].astype(np.int)
    for t in range(len(vlImgs_mask)):
        vlImgs_mask[t] = vlImgs_mask[t].astype(np.int)
    
    
    ## Box Koordinaten für die Bilder speichern
    tr_box_cord = []
    for t in range(len(trImgs_mask)):
        tr_box_cord.append(regionprops(trImgs_mask[t])[0].bbox)
    vl_box_cord = []
    for t in range(len(vlImgs_mask)):
        vl_box_cord.append(regionprops(vlImgs_mask[t])[0].bbox)
    
    ## Original Bilder beschneiden und nur noch die Bilder mít dem BB Boxen 
    trImgs_bb = []
    for t in range(len(trImgs)):
        props = tr_box_cord[t]
        trImgs_bb.append(trImgs[t][props[0]:props[2],props[1]:props[3]])
    vlImgs_bb = []
    for t in range(len(vlImgs)):
        props = vl_box_cord[t]
        vlImgs_bb.append(vlImgs[t][props[0]:props[2],props[1]:props[3]])
    
    return trImgs_bb, vlImgs_bb



def imgs_labels(label_list):
    label_list_new = label_list    
    for t in label_list:
        if re.search("Apple",t):
            label_list_new[label_list.index(t)] = 1
        if re.search("Aubergine",t):
            label_list_new[label_list.index(t)] = 2
        if re.search("Banana",t):
            label_list_new[label_list.index(t)] = 3
        if re.search("Carrots",t):
            label_list_new[label_list.index(t)] = 4
        if re.search("Cucumber",t):
            label_list_new[label_list.index(t)] = 5
        if re.search("Ginger",t):
            label_list_new[label_list.index(t)] = 6
        if re.search("Lemon",t):
            label_list_new[label_list.index(t)] = 7        
        if re.search("Melon",t):
            label_list_new[label_list.index(t)] = 8
        if re.search("Orange",t):
            label_list_new[label_list.index(t)] = 9
        if re.search("Pear",t):
            label_list_new[label_list.index(t)] = 10
        if re.search("Pepper",t):
            label_list_new[label_list.index(t)] = 11       
        if re.search("Pineapple",t):
            label_list_new[label_list.index(t)] = 12       
        if re.search("Potato",t):
            label_list_new[label_list.index(t)] = 13
        if re.search("Tomato",t):
            label_list_new[label_list.index(t)] = 14
        if re.search("Zucchini",t):
            label_list_new[label_list.index(t)] = 15
        if re.search("Watermelon",t):
            label_list_new[label_list.index(t)] = 16
    return np.array(label_list_new)
         