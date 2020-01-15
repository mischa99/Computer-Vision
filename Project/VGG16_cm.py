#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:14:49 2019

@author: 7dill
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:00:30 2019

@author: 7dill
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input,GlobalAveragePooling2D, BatchNormalization
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from sklearn.metrics import confusion_matrix, classification_report
import os
from skimage.io import imread
import random
import pickle
import pandas as pd
from sklearn.utils import resample

import itertools
from keras.models import load_model




FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/datasetx3"
FILEPATH_WEIGHTS= "/informatik1/students/home/9raudin/Desktop/CV/Project/"
weights_name="VGG16_x3weights.h5"

'''
IMG_SIZE = 224
CATEGORIES = ["Apple", "Aubergine", "Banana", "Carrots", "Cucumber", "Ginger", "Lemon", "Melon", "Orange", "Pear", "Pepper", "Pineapple", "Potato", "Tomato", "Watermelon", "Zucchini"]

training_data=[]
for category in CATEGORIES:  

    path = os.path.join(FILEPATH_DATA + "/val/" ,category)  # create path to food image folders
    class_num = CATEGORIES.index(category)  # get the classification labels
    i = 0
    for img in random.choice(os.listdir(path)): # iterate over each image per folder
        if (i==18): break
        try:
            img_array = imread(os.path.join(path,img))  # convert to array
            img_array = np.resize(img_array,(224,224,3))
            training_data.append([img_array,class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass
        i=i+1

print(len(training_data))


random.shuffle(training_data) 
for sample in training_data[:10]:
    print(sample[1])

X_train = []
y_train = []

for features,label in training_data:
    X_train.append(features)
    y_train.append(label)


#print(X_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))
#X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X_train = np.array(X_train)
y_train = np.array(y_train)
#print(X_train[0].shape)

#save feature array
pickle_out = open("X_val.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
#safe labels array
pickle_out = open("y_val.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("X_val.pickle","rb")
X_val = pickle.load(pickle_in)

pickle_in = open("y_val.pickle","rb")
y_val = pickle.load(pickle_in)

print(X_train.shape)
# concatenate our training data back together
y_train=y_train[:,np.newaxis]
X = np.concatenate([X_train, y_train], axis=1)

# separate minority and majority classes
apple = X[X.Class==0]

for i in list(range(0,15,1)):
    curr_class = X[X.Class==i]
    # upsample minority
    curr_class_upsampled = resample(curr_class,
                          replace=True, # sample with replacement
                          n_samples=len(apple), # match number in majority class
                          random_state=27) # reproducible results

    # combine majority and upsampled minority
    X = X + curr_class_upsampled
  
'''

'''
new_input=Input(shape=(224,224,3))
model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

#mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers
#x = GlobalAveragePooling2D()(model.output)
flat1 = Flatten()(model.output)
fc1 = Dense(256, activation='relu')(flat1)
#drop1 = Dropout(0.2)(fc1)
#batch_norm1=BatchNormalization(axis=1)(drop1)
#fc2 = Dense(256, activation='relu')(fc1)
#drop2 = Dropout(0.2)(fc2)
#batch_norm2=BatchNormalization(axis=1)(drop2)
output = Dense (16, activation= 'softmax')(fc1)

# define new model
model = Model(inputs=model.inputs, outputs=output)

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])

'''

train_datagen = ImageDataGenerator(
        rescale=1./255
        )

val_datagen = ImageDataGenerator(
        rescale=1./255
        )

test_datagen = ImageDataGenerator(
        rescale=1./255
        )


train_generator = train_datagen.flow_from_directory(
    directory=FILEPATH_DATA + "/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    
)

val_generator = val_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "/val/",
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=8,
        class_mode='categorical',
        shuffle=True,
        )

test_generator = test_datagen.flow_from_directory(
        directory= FILEPATH_DATA + "/test/",
        target_size=(224, 224),
        batch_size=8,
        class_mode=None, #only data, no labels
        shuffle=False, # keep data in order of files = labels
        )
   

'''
#CNN trainieren
history = model.fit_generator(train_generator, epochs=20, validation_data=val_generator, verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss',min_delta=0,patience=20),
                   ModelCheckpoint(filepath=FILEPATH_WEIGHTS+weights_name,
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

model.load_weights(weights_name, by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)

model.save('VGG16_1model.h5')
#test_imgs, test_labels = next(test_generator)
#test_labels = test_labels[:,0]
'''


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





#load model an weights
model = load_model('VGG16_1model.h5')
model.load_weights(weights_name, by_name=True)

#make predictions on test set
predictions = model.predict_generator(test_generator, verbose=1)

predicted_classes=np.argmax(predictions,axis=1) #use argmax to get highest value in each sequence = highest probability of class per test img
class_labels = list(test_generator.class_indices.keys())   

#arg 2 : np.round(predictions[:,0]
cm = confusion_matrix(test_generator.classes, predicted_classes)
np.set_printoptions(precision=2)
#disp = plot_confusion_matrix =(cm,cmap=plt.cm.Blues)

pdf= PdfPages('/informatik1/students/home/9raudin/Desktop/CV/Project/vgg16_1x3_cm.pdf')
# Plot non-normalized confusion matrix
fig1 = plt.figure()
plot_confusion_matrix(cm, classes=class_labels,
                      title='Confusion matrix, without normalization')
plt.show()
plt.close()
pdf.savefig(fig1) #add cm to pdf

# Plot normalized confusion matrix
fig2 = plt.figure()
plot_confusion_matrix(cm, classes=class_labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
plt.close()
pdf.savefig(fig2) #add cm to pdf
pdf.close()

'''
pdf = PdfPages('/informatik1/students/home/9raudin/Desktop/CV/Project/vgg16_1x3_cm.pdf')
fig1=plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title('Confusion matrix ')
plt.colorbar()
plt.close()
pdf.savefig(fig1) #add cm to pdf
pdf.close()
'''

report = classification_report(test_generator.classes, predicted_classes, target_names=class_labels)
print(report)    

'''
#create pdf to store loss and accuracy graphs
pdf = PdfPages('/informatik1/students/home/9raudin/Desktop/CV/Project/vgg16_x3.256db8_run2_loss_acc_graph.pdf')
#plot model loss
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
#plt.show() #show in window/terminal, doesn't work over ssh
plt.close()
pdf.savefig(fig1) #add loss figure to pdf

# plot model accuracy
fig2 = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.close()
pdf.savefig(fig2) #add acc figure to pdf

pdf.close() #close pdf after adding two plots

'''
'''
Batch Size 4, 12/15 Ep, val loss springt von 0,x auf 2,../5,.. und zurück. Ergebnis: 1.1669597625732422 0.7301006317138672

x2:
128-128: 95.6%
256-128: 97.8%

x3:
256: 98.2%
'''
