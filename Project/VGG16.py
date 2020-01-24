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

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications


FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/GroceryStoreDataset-V6_CNN"
FILEPATH_WEIGHTS= "/informatik1/students/home/9raudin/Desktop/CV/Project/"
weights_name="VGG16_V6weights.h5"
plot_name"XXXX"


  
new_input=Input(shape=(224,224,3))
model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

#mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers on top of VGG - Feature Extraxtions (-> add a top model)
#x = GlobalAveragePooling2D()(model.output)
flat1 = Flatten()(model.output)
fc1 = Dense(256, activation='relu')(flat1)
#drop1 = Dropout(0.2)(fc1)
#batch_norm1=BatchNormalization(axis=1)(drop1)
fc2 = Dense(256, activation='relu')(fc1)
#drop2 = Dropout(0.2)(fc2)
#batch_norm2=BatchNormalization(axis=1)(drop2)
output = Dense (16, activation= 'softmax')(fc2)


# define new model
model = Model(inputs=model.inputs, outputs=output)

#CNN komplieren
model.compile(loss='categorical_crossentropy',
optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        )

val_datagen = ImageDataGenerator(
        rescale=1./255,
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


#define some callbacks for training
early_stopping  = EarlyStopping(monitor='val_loss',min_delta=0,patience=20)
model_checkpoint= ModelCheckpoint(filepath=FILEPATH_WEIGHTS+weights_name,
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#CNN trainieren
history = model.fit_generator(train_generator, epochs=20, validation_data=val_generator, verbose=1,
        callbacks=[early_stopping,model_checkpoint])
    
model.load_weights(weights_name, by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)

model.save('VGG16_V6model.h5')

#create pdf to store loss and accuracy graphs
pdf =  PdfPages(FILEPATH_DATA + plot_name)
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
Batch Size 4, 12/15 Ep, val loss springt von 0,x auf 2,../5,.. und zurück. Ergebnis: 1.1669597625732422 0.7301006317138672

x2:
128-128: 95.6%
256-128: 97.8%

x3:
256: 98.2%
128: 98.2%
'''
