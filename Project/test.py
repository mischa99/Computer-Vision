#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:15:16 2019

@author: JanStrich
"""
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import scikitplot as skplt


skplt.metrics.plot_confusion_matrix(
        vlLabels,
        predictions,
        figsize=(16, 16),title="Confusion matrix", normalize = True)

plt.ylim(0, 16)
plt.xlim(0,16)
plt.show()