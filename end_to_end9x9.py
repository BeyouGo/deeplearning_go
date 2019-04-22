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
from keras.utils import np_utils
from betago.model_9x9 import KerasBot, HTTPFrontend
from betago.processor9x9 import SevenPlaneProcessor, SevenPlaneFileProcessor, SixPlaneProcessor, ThreePlaneProcessor, \
    TwoPlaneProcessor
# from betago.processor9x9 import ThreePlaneProcessor
import tensorflow as tf
from keras.models import model_from_yaml
from models import *
import matplotlib.pyplot as plt

import matplotlib


def show_graph(history,n_epochs,n_samples, n_batch ,model_name):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name+'_'+str(n_samples)+'_'+n_epochs+'_'+n_batch+'.png')
    plt.show()




num_samples = 12000 #1000
batch_size = 128 #128
nb_epoch = 20 #100

nb_classes = 9 * 9  # One class for each position on the board
go_board_rows, go_board_cols = 9, 9  # input dimensions of go board
nb_filters = 32  # number of convolutional filters to use
nb_pool = 2  # size of pooling area for max pooling
nb_conv = 3  # convolution kernel size


##### PROCESSOR

# SevenPlaneProcessor loads seven planes (doh!) of 19*19 data points, so we need 7 input channels
# processor = SevenPlaneProcessor(data_directory='data_9x9')
# processor = SixPlaneProcessor(data_directory='data_9x9')
processor = TwoPlaneProcessor(data_directory ='data_9x9')
# processor = ThreePlaneProcessor(data_directory='data_9x9')

##### Data Extraction

input_channels = processor.num_planes

# Load go data from 1000 KGS games and one-hot encode labels
X, y = processor.load_go_data_9x9(num_samples=2000,data_dir='data_9x9')

print("X LENGTH:",len(X), " ", len(X[0]))
X = X[:num_samples]
y = y[:num_samples]

X = X.astype('float32')
Y = np_utils.to_categorical(y, nb_classes)


##### MODEL
# model = model_3(input_channels=input_channels)
# model = model_padding_last_replaced(input_channels)
# model = model_E(input_channels)
model = model_E(input_channels)
#model.summary()

print(processor.num_planes)

# Fit model to data
history = model.fit(X, Y, batch_size=batch_size,epochs=nb_epoch, verbose=1)
filename = 'model-' + str(num_samples) + '-' + str(nb_epoch) + '-' + str(batch_size) + '.hd5'
model.save(filename)
model.summary()

show_graph(history,nb_epoch,num_samples,batch_size,"modelE")
# # Open web frontend
path = os.getcwd().replace('/examples', '')
webbrowser.open('file://' + path + '/ui/demoBot_9x9.html', new=2)

# Create a bot from processor and model, then serve it.
go_model = KerasBot(model=model, processor=processor)
graph = tf.get_default_graph()
go_server = HTTPFrontend(bot=go_model,graph=graph,port=8080)
webbrowser.open('http://0.0.0.0:8080/', new=2)
go_server.run()


