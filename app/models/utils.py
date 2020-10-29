import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_processing(data):
    temp_data = data["T (degC)"]
    temp_data = pd.DataFrame({"Date Time": data.index, "T (degC)":temp_data.values})
    temp_data = temp_data.set_index(["Date Time"])
    return temp_data

def data_scaling(data):
    temp_scaler = MinMaxScaler()
    temp_scaler.fit(data)
    normalized_temp = temp_scaler.transform(data)
    normalized_temp = pd.DataFrame(normalized_temp, columns=['T (degC)'])
    normalized_temp.index = data.index
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

def train_test_split(series):
    labels_col = 'T (degC)(t)'
    labels = series[labels_col]
    series = series.drop(['T (degC)(t)'], axis=1)
    X_train = series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
    X_valid = series['01.01.2015 00:00:00':'2017-01-01 00:00:00']
    Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
    Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']
    return X_train,Y_train,X_valid,Y_valid

def reshaping(X_train, X_valid):
    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    return X_train_series,X_valid_series