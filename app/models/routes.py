from flask import Blueprint, render_template, jsonify
import pandas as pd
import pickle

from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.regularizers import l1
from tensorflow.keras.models import load_model


# project related imports:
from app.models.utils import series_to_supervised,data_processing, data_scaling, train_test_split, reshaping

models = Blueprint('models', __name__)

'''
#model = pickle.load('simple_model.pkl', 'rb')
#with open('app/models/simple_model_pickle.pkl', 'rb') as file:
#    model = pickle.load(file)
'''

model = load_model('app/models/simple_model')

'''
# loading the model
from keras.models import model_from_yaml

with open('app/models/simple_model.yaml', 'r') as yaml_file:
    loaded_model_yaml = yaml_file.read()

    loaded_model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model.load_weights("y_model_weights.h5")
    model = loaded_model
'''

'''
    # predict
    pred_yaml = loaded_model.predict(x)
    pred_yaml = np.sqrt(mean_squared_error(pred_yaml, y))
    print("After loading score (RMSE): {}".format(pred_yaml))
'''


@models.route('/prediction', methods=['GET'])
def predict():
    data = pd.read_csv('app/data/climate_hour.csv', parse_dates=['Date Time'],index_col = 0, header=0)
    data = data.sort_values(['Date Time'])
    temp_data = data_processing(data)
    normalized_temp = data_scaling(temp_data)
    series = series_to_supervised(normalized_temp, window=144)
    X_train,Y_train,X_valid,Y_valid = train_test_split(series)
    X_train_reshaped, X_valid_reshaped = reshaping(X_train,X_valid)

    # Normalized predictions:
    lstm_train_pred = model.predict(X_train_reshaped)
    lstm_valid_pred = model.predict(X_valid_reshaped)

    print(lstm_train_pred)
    print(lstm_valid_pred)
    y_tr = Y_train.tolist()
    p_tr = lstm_train_pred.tolist()
    y_val = Y_valid.tolist()
    p_val = lstm_valid_pred.tolist()

    items = [   {'true_train': y_tr},
                {'pred_train':p_tr},
                {'true_val': y_val},
                {'pred_val': p_val} ]

    return jsonify(items)

'''    return jsonify({'true_train': Y_train.tolist(),
                    'pred_train':lstm_train_pred,
                    'true_val': Y_valid,
                    'pred_val': lstm_valid_pred})'''