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
    TwoPlaneProcessor, ThreePlaneOnesProcessor, FourPlaneOnesProcessor, EightPlaneProcessor
# from betago.processor9x9 import ThreePlaneProcessor
import tensorflow as tf
from keras.models import model_from_yaml

from models import *
import matplotlib.pyplot as plt
import numpy as np

import matplotlib


def show_graph(history, n_epochs, n_samples, n_batch, model_name, processorName):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(
        model_name + '_' + processorName + '_' + str(n_samples) + '_' + str(n_epochs) + '_' + str(n_batch) + '.png')
    plt.show()


num_samples = 1350000  # 1000
batch_size = 128  # 128
nb_epoch = 100  # 100

nb_classes = 9 * 9  # One class for each position on the board
go_board_rows, go_board_cols = 9, 9  # input dimensions of go board
nb_filters = 32  # number of convolutional filters to use
nb_pool = 2  # size of pooling area for max pooling
nb_conv = 3  # convolution kernel size

##### PROCESSOR

# SevenPlaneProcessor loads seven planes (doh!) of 19*19 data points, so we need 7 input channels
processor = SevenPlaneProcessor(data_directory='data_9x9')
# processor = SixPlaneProcessor(data_directory='data_9x9')
# processor = TwoPlaneProcessor(data_directory ='data_9x9')
# processor = ThreePlaneProcessor(data_directory='data_9x9')
# processor = ThreePlaneOnesProcessor(data_directory='data_9x9')
# processor = FourPlaneOnesProcessor(data_directory='data_9x9')
# processor = EightPlaneProcessor(data_directory='data_9x9')

##### Data Extraction

input_channels = processor.num_planes

# Load go data from 20000 KGS games and one-hot encode labels
Xall, yall = processor.load_go_data_9x9(num_samples=20000, data_dir='data_9x9')

print("X LENGTH:", len(Xall), " ", len(Xall[0]))
X = Xall[:num_samples]
y = yall[:num_samples]

X_test = Xall[num_samples:]
y_test = yall[num_samples:]
X_test = X_test[:1000]
y_test = y_test[:1000]

X = X.astype('float32')
Y = np_utils.to_categorical(y, nb_classes)

##### MODEL
# model = model_3(input_channels=input_channels)
# model = model_padding_last_replaced(input_channels)
model = model_G(input_channels)
# model = model_E(input_channels)
# model = model_3(input_channels)
# model = model_D(input_channels)
# model = model_C(input_channels)
# model.summary()

modelname = "model_G"
processorName = "7Plane_TF"

print(processor.num_planes)

# Fit model to data
history = model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.05)
filename = 'model-' + str(num_samples) + '-' + str(nb_epoch) + '-' + str(
    batch_size) + '-' + modelname + '-' + processorName + ".hd5"
model.save(filename)
model.summary()

show_graph(history, nb_epoch, num_samples, batch_size, modelname, processorName)

tops = [1, 5]
for top in tops:
    count_good = 0
    predictions = model.predict_on_batch(X_test)
    for i in range(0, len(predictions)):
        prediction = predictions[i]
        actual_move = int(y_test[i])

        prediction = np.array(prediction)
        # print(actual_move, ' - ', prediction.argsort()[-top:][::-1])

        if actual_move in prediction.argsort()[-top:][::-1]:
            count_good = count_good + 1
    print("top ", top, ": ", float(count_good) / len(predictions))

# # Open web frontend
path = os.getcwd().replace('/examples', '')
webbrowser.open('file://' + path + '/ui/demoBot_9x9.html', new=2)

# Create a bot from processor and model, then serve it.
go_model = KerasBot(model=model, processor=processor)
graph = tf.get_default_graph()
go_server = HTTPFrontend(bot=go_model, graph=graph, port=8080)
webbrowser.open('http://0.0.0.0:8080/', new=2)
go_server.run()

model.summary()
