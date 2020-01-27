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
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.callbacks import TensorBoard

'''
NEW******
Tensorboard Feature Plot and Compare different Models in one Graph
1) Specify NAME Variable of current model
2) Run model
3) after training finished, the data is being add to a new file called "logs"
4) To open the Board, after training, write in the terminal (at the project dir): tensorboard --logdir=logs/
-> gives a link to open the Board
'''


FILEPATH_DATA="/informatik1/students/home/9raudin/Desktop/CV/Project/DataX3_Final"
FILEPATH= "/informatik1/students/home/9raudin/Desktop/CV/Project/"

#Hyperparameters to optimize
dense_layers = [1,2]
layer_sizes = [32,64]  #already tried 128,256
dropout = [0.2,0.3,0.4,0.5]
activation = ['ELU(alpha=1.0)','LeakyReLU(alpha=0.1)'] #haven't tried yet, if trying -> model.add(activation)

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for i in range(len(dropout)):
            NAME = "{}-dense-{}-nodes-{}-dropout-{}".format(dense_layer, layer_size, dropout[i], int(time.time()))
            print(NAME)
            
            new_input=Input(shape=(224,224,3))
            model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=new_input)

            #mark loaded layers as not trainable
            for layer in model.layers:
                layer.trainable = False

            # add new classifier layers on top of VGG16 (-> add a top model)
            #x = GlobalAveragePooling2D()(model.output)
            flat1 = Flatten()(model.output)
            if(dense_layer==1):
                fc1 = Dense(layer_size, activation='relu')(flat1)
                drop1 = Dropout(dropout[i])(fc1)
                output = Dense (16, activation= 'softmax')(drop1)
                # define new model
                model = Model(inputs=model.inputs, outputs=output)
            if(dense_layer==2):
                fc1 = Dense(layer_size, activation='relu')(flat1)
                drop1 = Dropout(dropout[i])(fc1)
                fc2 = Dense(layer_size, activation='relu')(drop1)
                drop2 = Dropout(dropout[i])(fc2)
                output = Dense (16, activation= 'softmax')(drop2)
                # define new model
                model = Model(inputs=model.inputs, outputs=output)

            #CNN komplieren
            model.compile(loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.00017654638975080178 , beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy'])
       
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
            model_checkpoint= ModelCheckpoint(filepath=FILEPATH+"Weights-"+NAME+".h5",
                               monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            #reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

            #train CNN
            history = model.fit_generator(train_generator, epochs=20, validation_data=val_generator, verbose=1,
                    callbacks=[early_stopping,model_checkpoint,tensorboard])
                
            model.load_weights("Weights-"+NAME+".h5", by_name=True)

            val_loss, val_acc = model.evaluate_generator(val_generator,
             callbacks=None, max_queue_size=10, use_multiprocessing=False, verbose=1)
            print(val_loss, val_acc)

            model.save("Model-"+NAME+".h5")
            
            #create pdf to store loss and accuracy graphs
            pdf =  PdfPages(FILEPATH + NAME + ".pdf")
            fig1 = plt.figure()
            #plt.subplot(2,1,1) # num rows, num cols, plot nr
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.title(NAME+ " Model Loss")
            #plt.show() #show in window/terminal, doesn't work over ssh
            plt.close()
            pdf.savefig(fig1) #add loss figure to pdf

            # plot model accuracy
            fig2 = plt.figure()
            #plt.subplot(2,1,2)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.title(NAME + " Model Accuracy")
            #plt.show()
            plt.close()
            pdf.savefig(fig2) #add acc figure to pdf

            pdf.close() #close pdf after adding two plots
            

    '''
    256-0.3-256-0.2
    Adam learning_rate=0.0001
    -> 86,84 %
    gleiche Daten mit BN -> 81,9 %, lr sinkt langsamer

    '''
