#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:05:43 2020

@author: 9raudin
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.utils import np_utils, plot_model

import plot_confusion_matrix

FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V6_CNN"
weights_name="VGG16_V6weights.h5"
model_name='VGG16_V6model.h5'
cm_name='/vgg16_1x3_cm.pdf'

test_datagen = ImageDataGenerator(
        rescale=1./255,
        )

test_generator = test_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "/test/",
        target_size=(224, 224),
        batch_size=8,
        class_mode=None, #only data, no labels
        shuffle=False, # keep data in order of files = labels
        )

#load model and weights
model = load_model(model_name)
model.load_weights(weights_name, by_name=True)

#make predictions on test set
predictions = model.predict_generator(test_generator, verbose=1)

predicted_classes=np.argmax(predictions,axis=1) #use argmax to get highest value in each sequence = highest probability of class per test img
class_labels = list(test_generator.class_indices.keys()) #get true labels   

cm = confusion_matrix(test_generator.classes, predicted_classes)
np.set_printoptions(precision=2)

pdf= PdfPages(FILEPATH_DATA + cm_name)
# Plot non-normalized confusion matrix
fig1 = plt.figure()
plot_confusion_matrix(cm, classes=class_labels,
                      title='Confusion matrix, without normalization')
pdf.savefig(fig1) #add cm to pdf

# Plot normalized confusion matrix
fig2 = plt.figure()
plot_confusion_matrix(cm, classes=class_labels, normalize=True,
                      title='Normalized confusion matrix')
pdf.savefig(fig2) #add cm to pdf
pdf.close() #pdf done

report = classification_report(test_generator.classes, predicted_classes, target_names=class_labels)
print(report)    
