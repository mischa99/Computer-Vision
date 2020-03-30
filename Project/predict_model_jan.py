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
import itertools

from sklearn.metrics import confusion_matrix, classification_report
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Sequential
from keras.utils import np_utils, plot_model


import scikitplot as skplt


### meine ursprüngliche CM methode
def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




###FILEPATH ÄNDERN
FILEPATH_DATA="/Users/mikhail/Desktop/PraktikumCV/Project/"
weights_name="Weights-2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772.h5"
model_name='Model-2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772.h5'
cm_name='vgg16-cm-2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772.h5'

test_datagen = ImageDataGenerator(
        rescale=1./255,
        )

test_generator = test_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "DataX3_Final/test/",
        target_size=(224, 224),
        batch_size=1,
        class_mode=None, #only data, no labels
        shuffle=False, # keep data in order of files = labels
        )

model = load_model(model_name)
model.load_weights(weights_name, by_name=True, compile = False)


#make predictions on test set
predictions = model.predict_generator(test_generator, verbose=1, steps=122)

predicted_classes=np.argmax(predictions,axis=1) #use argmax to get highest value in each sequence = highest probability of class per test img
class_labels = list(test_generator.class_indices.keys()) #get true labels   

cm = confusion_matrix(test_generator.classes, predicted_classes)
np.set_printoptions(precision=2)

pdf= PdfPages(FILEPATH_DATA + cm_name)
# Plot non-normalized confusion matrix
fig1 = plt.figure()
#plot_confusion_matrix(cm, classes=class_labels,
#  title='Confusion matrix, without normalization')


#### dein Confusion Matrix Script aus git
skplt.metrics.plot_confusion_matrix(
        class_labels,
        predicted_classes,
        figsize=(16, 16),title="Confusion matrix", normalize = True)

plt.ylim(0, 16)
plt.xlim(0,16)
plt.show()

pdf.savefig(fig1) #add cm to pdf
#pdf.close()

# Plot normalized confusion matrix
fig2 = plt.figure()
plot_confusion_matrix(cm, classes=class_labels, normalize=True,
                      title='Normalized confusion matrix')
pdf.savefig(fig2) #add cm to pdf
pdf.close() #pdf done


report = classification_report(test_generator.classes, predicted_classes, target_names=class_labels)
print(report)

'''
###PRINT WRONG PRED IMG
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
     
    original = imread('{}/{}'.format(FILEPATH_DATA + "DataX3_Final/test/",fnames[errors[i]]))
    fig = plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.close()
    pdf.savefig(fig)
    
pdf.close()
'''  
