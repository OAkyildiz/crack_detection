#!/usr/bin/env python
import keras.layers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import *
from keras.constraints import maxnorm

from keras import backend as K

key={'nc':'number of classes',
     'b':'batch_size',
     'h':'input height/input_shape[0]',
     'w':'input width/input_shape[1]',
     'k':'kernel_size',
     'fn':'activation(s)',
     }
def help():
    print(key)
# keras.layers.convolutional.Conv2D(filters, kernel_size,
#                                   strides=(1, 1), padding='valid',
#                                   data_format=None, dilation_rate=(1, 1),
#                                   activation=None, use_bias=True,
#                                   kernel_initializer='glorot_uniform',
#                                   bias_initializer='zeros', kernel_regularizer=None,
#                                   bias_regularizer=None, activity_regularizer=None,
#                                   kernel_constraint=None, bias_constraint=None)
def cifar_network(nc,b=100,h=32,w=32, k=3, fn=['relu','relu','relu','softmax']):
    model = Sequential()
    model.add(Conv2D(b, (k, k), input_shape=(h, w, 3), padding='same', activation=fn[0], kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(b, (k, k), activation=fn[1], padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation=fn[2], kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(nc, activation=fn[3]))

    return model

# Is there padding in Zhang's network?
# it is losing 4,4 then 3 pixels for each mp->conv
def zhang_network(nc,b=48,h=99,w=99,fn='relu'):
    #add dropouts
    model = Sequential()
    model.add(Conv2D(b, (4, 4), input_shape=(96, 96, 3)))
    model.add(Conv2D(b, (4, 4), input_shape=(96, 96, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #flatten here?
    model.add(Conv2D(b, (5, 5), input_shape=(44, 44, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(b, (3, 3), input_shape=(18, 18, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(b, (4, 4), input_shape=(6, 6, 3), activation=fn, kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(nc, activation='softmax'))

    return model
