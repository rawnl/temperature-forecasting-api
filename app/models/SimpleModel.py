import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.regularizers import l1
from keras.models import load_model


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import pickle

# Set seeds to make the experiment more reproducible.
import tensorflow as tf
from numpy.random import seed

tf.random.set_seed(1)
seed(1)

def data_processing(data):
    temp_data = data['T (degC)']
    temp_data = pd.DataFrame({'Date Time': data.index, 'T (degC)': temp_data.values})
    temp_data = temp_data.set_index(['Date Time'])
    temp_data.head()

    temp_scaler = MinMaxScaler()
    temp_scaler.fit(temp_data)
    normalized_temp = temp_scaler.transform(temp_data)

    normalized_temp = pd.DataFrame(normalized_temp, columns=['T (degC)'])
    normalized_temp.index = temp_data.index
    normalized_temp.head()

    return normalized_temp

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg

def split_data(series):
    labels_col = 'T (degC)(t)'
    labels = series[labels_col]
    series = series.drop(['T (degC)(t)'], axis=1)

    X_train = series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
    X_valid = series['01.01.2015 00:00:00':'2017-01-01 00:00:00']
    Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
    Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']

    return X_train, X_valid, Y_train, Y_valid

def reshape_data(X_train, X_valid):
    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))

    return X_train_series, X_valid_series

def build_model(input_shape):

    model_lstm = Sequential()
    model_lstm.add(LSTM(25, activation='sigmoid', input_shape=input_shape))  # , activity_regularizer=l1(0.001)
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mae', optimizer=opt, metrics=['mse'])
    model_lstm.summary()

    return model_lstm


window = 144
batch = 64
lr = 0.01
opt = optimizers.Adam(lr=lr)
epoch = 1
string = '{}-WIND{}-OPT{}-LR{}-EP{}-BAT{}'.format(int(time.time()), window, opt, lr, epoch, batch)
model_name = "Simple-LSTM-Model-{}".format(string)


data = pd.read_csv('../../app/data/climate_hour.csv', parse_dates=['Date Time'], index_col=0, header=0)
data = data.sort_values(['Date Time'])
data.head()

normalized_temp = data_processing(data)
series = series_to_supervised(normalized_temp, window=window)
X_train, X_valid, Y_train, Y_valid = split_data(series)
X_train_series, X_valid_series = reshape_data(X_train, X_valid)
model_lstm = build_model(input_shape=(X_train_series.shape[1], X_train_series.shape[2]))

# Training :
lstm_history = model_lstm.fit(X_train_series, Y_train,
                                validation_data=(X_valid_series, Y_valid),
                                epochs=epoch, verbose=1,
                                batch_size=batch)


# save model  structure to YAML (no weights)
model_yaml = model_lstm.to_yaml()
with open("simple_model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# save the model weights separatly
model_lstm.save_weights('y_model_weights.h5')
