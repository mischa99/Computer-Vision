#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:00:30 2019

@author: Mikhail Raudin, Timm Dill
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras import applications



FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/DataX3_Final"
FILEPATH= "/informatik1/students/home/9raudin/Desktop/CV/Project/"
NAME ="2-Dense-256-nodes-0.3_0.2-dropout-unfreeze-1580308772"
  
new_input=Input(shape=(224,224,3))
model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

#mark loaded layers as not trainable, unfreeze last conv block of vgg16
for layer in model.layers[:-4]:
    layer.trainable = False
    
# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)

# add new classifier layers on top of VGG - Feature Extraxtions (-> add a top model)
flat1 = Flatten()(model.output)

fc1 = Dense(256)(flat1)
act1 = LeakyReLU(alpha=0.1)(fc1)
drop1 = Dropout(0.3)(act1)

fc2 = Dense(256)(drop1)
act2 = LeakyReLU(alpha=0.1)(fc2)
drop2 = Dropout(0.2)(act2)

output = Dense (16, activation= 'softmax')(drop2)

# define new model
model = Model(inputs=model.inputs, outputs=output)

#compile model
model.compile(loss='categorical_crossentropy',
optimizer=Adam(learning_rate=0.00017654638975080178, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])

#define ImageDataGenerator to let image data flow from directory
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

model.load_weights("Weights-"+NAME+".h5", by_name=True) #load fc-layers weights to only train unfrozen vgg16 block

#define some callbacks for training
early_stopping  = EarlyStopping(monitor='val_loss',min_delta=0,patience=5)
model_checkpoint= ModelCheckpoint(filepath="Weights-"+NAME+".h5",
        monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=0)

#CNN trainieren
history = model.fit_generator(train_generator, epochs=20, validation_data=val_generator, verbose=1,
        callbacks=[early_stopping,model_checkpoint,tensorboard])
    
model.load_weights("Weights-"+NAME+".h5", by_name=True)

val_loss, val_acc = model.evaluate_generator(val_generator,
 callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
print(val_loss, val_acc)

#model.save("Model-"+NAME+".h5")

#create pdf to store loss and accuracy graphs
pdf =  PdfPages(FILEPATH + NAME + ".pdf")
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

