from flask import Blueprint, render_template, jsonify, request
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import yaml
import os
import json
from keras.models import model_from_yaml
import tensorflow.keras 
from tensorflow.keras import optimizers
from app import db, bcrypt, config

#from tensorflow.python.keras.models import Sequential

from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout
#from tensorflow.keras.regularizers import l1
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
def load_yaml_model(Simple=True):

    # load YAML and create model
    if Simple :
        yaml_file = open('app/models/simple_model.yaml', 'r')
    else:
        yaml_file = open('multi_step_model.yaml', 'r')

    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("y_model_weights.h5")
    print("Loaded model from disk")
    
    return load_model

def model_construction(shape=None):
    model_lstm = Sequential()
    model_lstm.add(LSTM(25, activation='sigmoid', input_shape=shape)) #, activity_regularizer=l1(0.001)
    model_lstm.add(Dense(1)) 
    #model_lstm.summary()

    return model_lstm

def model_loading(Simple=True):
    if Simple :
        model = tensorflow.keras.models.load_model('app/models/simple_model')
    else :
        model = tensorflow.keras.models.load_model('app/models/multi_model')
    return model

@models.route('/train', methods=['POST'])
def train():

    mdl = request.form.get('model')
    wind = request.form.get('window')
    opt = request.form.get('optimizer')
    lr = request.form.get('lr')
    btch = request.form.get('batch')
    ep = request.form.get('epochs')
    
    print('model:{} window:{} optimizer:{} lr:{} btch{} ep{}'.format(mdl, wind, opt, lr, btch, ep))
    
    #error : app is not defined
    #UPLOAD_FOLDER = 'app/data/uploads'
    csv_file = request.files['csv-file']
    filename = secure_filename(csv_file.filename)
    filename = os.path.join(config.UPLOAD_FOLDER, filename)
    csv_file.save(filename)

    #csv_file.save(os.path.join(UPLOAD_FOLDER, filename))
    
    '''
    if mdl == "simple":
        new_model = tensorflow.keras.models.load_model("app/models/new_simple_model.h5")
    else:
        new_model = tensorflow.keras.models.load_model("app/models/new_simple_model.h5")
    '''
    
    data = pd.read_csv(filename, index_col = 0, header=0, encoding= 'unicode_escape')#, encoding= 'unicode_escape' #,skiprows=5,nrows=200) #parse_dates=['Date Time'],

    data = data.sort_values(['Date Time'])
    print(data.columns)
    print(data.head)

    temp_data = data_processing(data)    
    normalized_temp = data_scaling(temp_data)
    series = series_to_supervised(normalized_temp, window=int(wind))

    row_count = sum(1 for line in open(filename)) - 1 # adds header    
    
    labels_col = 'T (degC)(t)'
    labels = series[labels_col]
    series = series.drop(['T (degC)(t)'], axis=1)
    
    X_train = series[:160] 
    X_valid = series[160:]
    Y_train = labels[:160]
    Y_valid = labels[160:]


    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1)) 

    #X_train,Y_train,X_valid,Y_valid = train_test_split(series)
    #X_train_reshaped, X_valid_reshaped = reshaping(X_train,X_valid)
    
    shape=(X_train_series.shape[1], X_train_series.shape[2])
    print(shape)
    model = model_construction(shape=shape)

    if opt=="Adam":
        optimizer = optimizers.Adam(lr=float(lr))
    elif opt=="RMSprop":
         optimizer = optimizers.RMSprop(lr=float(lr))
    elif opt == "SGD":
        optimizer = optimizers.SGD(lr=float(lr))   

    model.compile(loss='mae', optimizer=optimizer, metrics=['mse']) 

    history = model.fit(X_train_series, Y_train,
                    validation_data=(X_valid_series, Y_valid),
                    epochs=int(ep), verbose=1, batch_size=int(btch)) 
    #model fit
    lstm_train_pred = model.predict(X_train_series)
    lstm_valid_pred = model.predict(X_valid_series)

    print(Y_train)
    print(lstm_train_pred)

    items = [   {'true_train': Y_train.tolist()},
                {'pred_train':lstm_train_pred.tolist()},
                {'true_val': Y_valid.tolist()},
                {'pred_val': lstm_valid_pred.tolist()} ]
    
    predictions = json.dumps(items)
    return predictions, history, row_count 

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