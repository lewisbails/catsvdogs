# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:29:03 2018

@author: Hew
"""

#Dependencies
from __future__ import print_function

import tensorflow as tf
from keras import callbacks,optimizers,Model,models,layers
from keras.layers import Dense,GlobalAveragePooling2D,Conv2D,MaxPooling2D,Flatten,Activation
from keras.preprocessing import image
import numpy as np
import scipy.ndimage

#Hyperparameters
training_iters = 50
batch_size = 512

#BUild model
pred = models.Sequential()

pred.add(Conv2D(32, kernel_size=(10,10),strides=(5,5),
                input_shape=(256,256,3),
                activation='relu'))
pred.add(MaxPooling2D(pool_size=(5,5),strides=(5,5)))
pred.add(Conv2D(64,(5,5),activation='relu'))
pred.add(MaxPooling2D(pool_size=(2,2)))
pred.add(Flatten())
pred.add(Dense(2))
pred.add(Activation(tf.nn.softmax))
print("CNN built.")
pred.compile(optimizer='adamax',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
print("CNN compiled.")
#Train model
print("Training...")

# Training Data generator
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(256,256),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['cats','dogs'])

# Validation Data Generator
valid_datagen = image.ImageDataGenerator(
        rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
        'data/valid',
        target_size=(256,256),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['cats','dogs'])

# Test Data Generator
test_generator = valid_generator

# Train model
pred.fit_generator(train_generator,
                   steps_per_epoch=23000/batch_size,
                   epochs=5,
                   validation_data=valid_generator,
                   validation_steps=2000/batch_size,
                   verbose=1)

pred.save_weights("cnn.h5")

# Evaluate model
score = pred.evaluate_generator(test_generator,verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

