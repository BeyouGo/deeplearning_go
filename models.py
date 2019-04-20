from __future__ import print_function
import os
import webbrowser

import yaml
from keras.engine import Layer
from keras.layers import LSTM, Conv2DTranspose, Conv2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.utils import np_utils
from betago.model_9x9 import KerasBot, HTTPFrontend
from betago.processor9x9 import SevenPlaneProcessor
import tensorflow as tf
from keras.models import model_from_yaml


def model_padding_last_replaced(input_channels, pretrained_fixed=True):
    nb_classes = 9 * 9  # One class for each position on the board
    go_board_rows, go_board_cols = 9, 9  # input dimensions of go board
    # nb_filters = 32  # number of convolutional filters to use
    # nb_pool = 2  # size of pooling area for max pooling
    # nb_conv = 3  # convolution kernel size
    # padding = ((5, 5), (5, 5))
    padding = ((0,10), (0,10))
    #################################

    bot_name = 'demo'
    model_file = 'model_zoo/' + bot_name + '_bot.yml'
    weight_file = 'model_zoo/' + bot_name + '_weights.hd5'

    with open(model_file, 'r') as f:
        yml = yaml.load(f)
        pretrained_model = model_from_yaml(yaml.dump(yml))

        # Note that in Keras 1.0 we have to recompile the model explicitly
        pretrained_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        pretrained_model.load_weights(weight_file)
        # Remove the last 2 layers dense x => 19*19 and activation (softmax)
#        pretrained_model.summary()
        pretrained_model.pop()
        pretrained_model.pop()
#        pretrained_model.summary()

    #################################

    model = Sequential()
    model.add(ZeroPadding2D(input_shape=(input_channels, go_board_rows, go_board_cols),
                            padding=padding,
                            data_format='channels_first'))

    model.add(pretrained_model)
    if pretrained_fixed:
        # Set pretrained as not updatable
        model.layers[-1].trainable = False
    else:
        model.layers[-1].trainable = True

    model.add(Dropout(0.5))
    # Add a new softmax layer to replace the one removed from the pretrained model
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model



def model2(input_channels):
    nb_classes = 9 * 9  # One class for each position on the board
    go_board_rows, go_board_cols = 9, 9  # input dimensions of go board
    nb_filters = 32  # number of convolutional filters to use
    nb_pool = 2  # size of pooling area for max pooling
    nb_conv = 3  # convolution kernel size

    #################################

    bot_name = 'demo'
    model_file = 'model_zoo/' + bot_name + '_bot.yml'
    weight_file = 'model_zoo/' + bot_name + '_weights.hd5'

    with open(model_file, 'r') as f:
        yml = yaml.load(f)
        pretrained_model = model_from_yaml(yaml.dump(yml))

        # Note that in Keras 1.0 we have to recompile the model explicitly
        pretrained_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        pretrained_model.load_weights(weight_file)

    #################################

    model = Sequential()
    model.add(pretrained_model)
    model.layers[-1].trainable = False
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


def model_3():
    nb_classes = 9 * 9  # One class for each position on the board
    go_board_rows, go_board_cols = 9, 9  # input dimensions of go board
    nb_filters = 32  # number of convolutional filters to use
    nb_pool = 2  # size of pooling area for max pooling
    nb_conv = 3  #
    input_channels = 3

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid',
                     input_shape=(input_channels, go_board_rows, go_board_cols),
                     data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model