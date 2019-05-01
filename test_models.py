from __future__ import print_function
import os
import webbrowser
# import matplotlib as plt
import yaml
from keras.engine import Layer
from keras.layers import LSTM, Conv2DTranspose
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from betago.model_9x9 import KerasBot, HTTPFrontend
from betago.processor9x9 import SevenPlaneProcessor, SevenPlaneFileProcessor, SixPlaneProcessor, ThreePlaneProcessor, \
    TwoPlaneProcessor, EightPlaneProcessor
# from betago.processor9x9 import ThreePlaneProcessor
import tensorflow as tf
from keras.models import model_from_yaml
from models import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib


num_samples = 120000 #1000
batch_size = 128 #128
nb_epoch = 20 #100

nb_classes = 9 * 9  # One class for each position on the board
go_board_rows, go_board_cols = 9, 9  # input dimensions of go board
nb_filters = 32  # number of convolutional filters to use
nb_pool = 2  # size of pooling area for max pooling
nb_conv = 3  # convolution kernel size

model_weight_filename = "model-1350000-50-128-model_E-EightPlaneOnes.hd5"
##### PROCESSOR

# SevenPlaneProcessor loads seven planes (doh!) of 19*19 data points, so we need 7 input channels
processor = SevenPlaneProcessor(data_directory='data_9x9')
# processor = EightPlaneProcessor(data_directory='data_9x9')
# processor = SixPlaneProcessor(data_directory='data_9x9')
# processor = TwoPlaneProcessor(data_directory ='data_9x9')
# processor = ThreePlaneProcessor(data_directory='data_9x9')

##### Data Extraction

input_channels = processor.num_planes
#
# # Load go data from 1000 KGS games and one-hot encode labels
# X, y = processor.load_go_data_9x9(num_samples=2000,data_dir='data_9x9')
#
# print("X LENGTH:",len(X), " ", len(X[0]))
# X = X[num_samples:]
# # X = X[:1000]
# y = y[num_samples:]
# # y = y[:1000]
#
# X = X.astype('float32')
# Y = np_utils.to_categorical(y, nb_classes)


##### MODEL
# model = model_C(input_channels=input_channels)
# model = model_padding_last_replaced(input_channels)
#
model = model_4(input_channels)
# model = model_E(input_channels)
model.build((8,input_channels,9,9))
model.summary()



plot_model(model,to_file='model3_schema.png',show_shapes=True,show_layer_names=True)
#
# model.load_weights(model_weight_filename)
#
# top = 5
# print(len(y))
#
# count_good = 0
# predictions = model.predict_on_batch(X)
# for i in range(0, len(predictions)):
#     prediction = predictions[i]
#     actual_move = int(y[i])
#
#     prediction = np.array(prediction)
#     print(actual_move,' - ', prediction.argsort()[-top:][::-1] )
#
#
#     if actual_move in prediction.argsort()[-top:][::-1]:
#         count_good = count_good + 1
#
#
# print(float(count_good)/len(predictions) )

#
# # Fit model to data
# history = model.fit(X, Y, batch_size=batch_size,epochs=nb_epoch, verbose=1)
# filename = 'model-' + str(num_samples) + '-' + str(nb_epoch) + '-' + str(batch_size) + '.hd5'
# model.save(filename)
# model.summary()

# show_graph(history,nb_epoch,num_samples,batch_size,"modelE")
# # Open web frontend
path = os.getcwd().replace('/examples', '')
webbrowser.open('file://' + path + '/ui/demoBot_9x9.html', new=2)

# Create a bot from processor and model, then serve it.
go_model = KerasBot(model=model, processor=processor)
graph = tf.get_default_graph()
go_server = HTTPFrontend(bot=go_model,graph=graph,port=8080)
webbrowser.open('http://0.0.0.0:8080/', new=2)
go_server.run()


