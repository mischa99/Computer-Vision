# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:43:23 2019

@author: Jan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:51:30 2019

@author: Jan
"""

#always do this before working with keras!
import numpy as np
np.random.seed(123)# um die Gewichte immer gleichzufaellig zu initialisieren

#import tensorflow as tf
from tensorflow import set_random_seed
#tf.random.set_seed(123)# -''-
set_random_seed(123)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD


data = np.load("trainingsDatenFarbe2.npz")
data2 = np.load("validierungsDatenFarbe2.npz")


trImgs = data['data']
trLabels = data['labels']
vlImgs = data2['data']
vlLabels = data2['labels']



#### Merkmale berechnen ------------------------------------------
trmeans = []
for t in list(range(0,60,1)):
   trmeans.append(np.mean(trImgs[t,:,:], axis = (0,1)))
   
vlmeans = []
for t in list(range(0,30,1)):
   vlmeans.append(np.mean(vlImgs[t,:,:], axis = (0,1)))

trstds = []
for t in list(range(0,60,1)):
   trstds.append(np.std(trImgs[t,:,:], axis = (0,1)))
   
vlstds = []
for t in list(range(0,30,1)):
   vlstds.append(np.std(vlImgs[t,:,:], axis = (0,1)))   

tr_descriptors = np.concatenate((trmeans,trstds), axis=1) #axis=1 -> column wise, else row
vl_descriptors = np.concatenate((vlmeans,vlstds), axis=1)

tr_descriptors=tr_descriptors.astype('float32')
vl_descriptors=vl_descriptors.astype('float32')
#print(tr_descriptors.dtype)   #how to check if successfull

arr=[1,4,8]
y=0
for x in arr:
    trLabels=np.where(trLabels==x,y,trLabels)
    vlLabels=np.where(vlLabels==x,y,vlLabels)
    y+=1
    
# Label zum umformatieren 
Y_train = np_utils.to_categorical(trLabels, 3)
Y_test = np_utils.to_categorical(vlLabels, 3)

model = Sequential()
model.add(Dense(8, activation='relu', name='fc1', input_shape=(6,)))
model.add(Dense(8, activation='relu', name='fc2')) 
model.add(Dense(3, activation='softmax')) 

model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.000005, momentum=0.9), metrics=['accuracy'])

model.fit(tr_descriptors, Y_train, batch_size=1, epochs=500, verbose=1)

score = model.evaluate(vl_descriptors, Y_test, verbose=1)
print(score)


'''
Die Ergebnisse sind kaum besser im vgl. zum Nächsten-Nachbar-Klassifikator:
 bei Batch Size 1 , Epochen=500 -> 63,33% , loss -> 0,98  (53% bei Nächster_Nachbar)
 bei Batch Size 1 , Epochen=1000 -> 63,33% ,loss -> 1,04
 bei Batch Size 1 , Epochen=2000 -> 60,00% , loss -> 1,24
 bei Batch Size 2 , Epochen=2000 -> 60,00% , loss -> 1,00
 bei Batch Size 1 , Epochen=5000 -> 60,00% , loss -> 1,24
 bei Batch Size 10 , Epochen=5000 -> 53,33% , loss -> 5,67
'''

