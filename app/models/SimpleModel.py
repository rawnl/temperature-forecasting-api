import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import pickle

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import l1

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# project related imports:
from app.models.utils import series_to_supervised

#data processing:
data = pd.read_csv('../data/climate_hour.csv', parse_dates=['Date Time'],index_col = 0, header=0)
data = data.sort_values(['Date Time'])

temp_data = data['T (degC)']
temp_data = pd.DataFrame({'Date Time': data.index, 'T (degC)':temp_data.values})
temp_data = temp_data.set_index(['Date Time'])
temp_data.head()

temp_scaler = MinMaxScaler()
temp_scaler.fit(temp_data)
normalized_temp = temp_scaler.transform(temp_data)

normalized_temp = pd.DataFrame(normalized_temp, columns=['T (degC)'])
normalized_temp.index = temp_data.index

window = 144
batch = 32
lr = 0.01
opt = optimizers.Adam(lr=lr)
epoch = 20
string = '{}-WIND{}-OPT{}-LR{}-EP{}-BAT{}'.format(int(time.time()),window,opt,lr,epoch,batch)

series = series_to_supervised(normalized_temp, window=window)
labels_col = 'T (degC)(t)'
labels = series[labels_col]
series = series.drop(['T (degC)(t)'], axis=1)
X_train = series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
X_valid = series['01.01.2015 00:00:00':'2017-01-01 00:00:00']
Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']

#Reshape :
X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))

# model name :
model_name = "Simple-LSTM-Model-{}".format(string)

# The model :
model_lstm = Sequential()
model_lstm.add(LSTM(25,activation='sigmoid',input_shape=(X_train_series.shape[1], X_train_series.shape[2])))  # , activity_regularizer=l1(0.001)
model_lstm.add(Dense(1))
model_lstm.compile(loss='mae', optimizer=opt, metrics=['mse'])
model_lstm.summary()

#Training :
start = time.time()
lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epoch, verbose=1, batch_size=batch)
end = time.time()
execution_time = end - start
print(execution_time)

#Ploting loss history :
lstm_train_loss = lstm_history.history['loss']
lstm_test_loss = lstm_history.history['val_loss']
epoch_count = range(1, len(lstm_train_loss)+1)
plt.plot(epoch_count, lstm_train_loss)
plt.plot(epoch_count, lstm_test_loss)
plt.title('loss history')
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()
#plt.savefig('loss-history-plot-{}.png'.format(string))

# Normalized predictions:
lstm_train_pred = model_lstm.predict(X_train_series)
lstm_valid_pred = model_lstm.predict(X_valid_series)

# calculate RMSE on normalized data :
n_train_rmse = np.sqrt(mean_squared_error(Y_train, lstm_train_pred))
n_val_rmse = np.sqrt(mean_squared_error(Y_valid, lstm_valid_pred))
print('Train rmse (avec normalisation):', n_train_rmse)
print('Validation rmse (avec normalisation):', n_val_rmse)

# calculate MAE on normalized data:
n_train_mae = mean_absolute_error(Y_train, lstm_train_pred)
n_val_mae = mean_absolute_error(Y_valid, lstm_valid_pred)
print('Train mae (avec normalisation):', n_train_mae)
print('Validation mae (avec normalisation):', n_val_mae)

# serialize weights to HDF5
model_lstm.save_weights("model.h5")
print("Saved")

#Create data frame for predictions (normalized):
normalized_lstm_predictions = pd.DataFrame(Y_valid.values, columns=['Temperature'])
normalized_lstm_predictions.index = X_valid.index
normalized_lstm_predictions['Predicted Temperature'] = lstm_valid_pred

#scale back data:
y_val_inv_lstm = temp_scaler.inverse_transform(Y_valid.values.reshape(-1,1))
pred_lstm_dataset = lstm_valid_pred.reshape(17471)
y_pred_inv_lstm = temp_scaler.inverse_transform(pred_lstm_dataset.reshape(-1,1))

#Create data frame for predictions (C°):
lstm_predictions = pd.DataFrame(y_val_inv_lstm, columns=['Temperature (C°)'])
lstm_predictions.index = Y_valid.index
lstm_predictions['Predicted Temperature(C°)'] = y_pred_inv_lstm

# calculate MAE on data (C°):
val_mae = mean_absolute_error(y_val_inv_lstm, y_pred_inv_lstm)
val_rmse = np.sqrt(mean_squared_error(y_val_inv_lstm, y_pred_inv_lstm))
print('Validation mae (sans normalisation):', val_mae)
print('Validation rmse (sans normalisation):', val_rmse)
