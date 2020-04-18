#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:05:43 2020

@author: Mikhail Raudin, Timm Dill
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import itertools

from sklearn.metrics import classification_report
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Sequential
from keras.utils import np_utils, plot_model

FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/DataX3_Final/test"
weights_name="Weights-2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772.h5"
model_name='Model-2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772.h5'
cm_name='vgg16-cm-2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772.pdf'

test_datagen = ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        directory= FILEPATH_DATA,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None, #only data, no labels
        shuffle=False, # keep data in order of files = labels
        )

model = load_model(model_name)
model.load_weights(weights_name, by_name=True)


#make predictions on test set
predictions = model.predict_generator(test_generator, verbose=1, steps=122)

predicted_classes=np.argmax(predictions,axis=1) #use argmax to get highest value in each sequence = highest probability of class per test img
class_labels = list(test_generator.class_indices.keys()) #get true labels   

report = classification_report(test_generator.classes, predicted_classes, target_names=class_labels)
print(report)


###PRINT WRONGLY PREDICTED IMAES
# Get the filenames from the generator
fnames = test_generator.filenames
 
# Get the ground truth from generator
ground_truth = test_generator.classes
 
# Get the label to class mapping from the generator
label2index = test_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))
 
pdf= PdfPages(FILEPATH_DATA + "Wrong Predictions Images")
# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]
     
    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])
     
    original = imread('{}/{}'.format(FILEPATH_DATA + "/",fnames[errors[i]]))
    fig = plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.close()
    pdf.savefig(fig)
    
pdf.close()
