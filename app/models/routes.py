from flask import Blueprint, render_template, jsonify
import pandas as pd
import pickle
from keras.models import load_model

# project related imports:
from app.models.utils import series_to_supervised,data_processing, data_scaling, train_test_split, reshaping

models = Blueprint('models', __name__)

#model = pickle.load('simple_model.pkl', 'rb')
#with open('app/models/simple_model_pickle.pkl', 'rb') as file:
#    model = pickle.load(file)

model = load_model('pickle_model.sav')

@models.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv('../data/climate_hour.csv', parse_dates=['Date Time'],index_col = 0, header=0)
    data = data.sort_values(['Date Time'])
    temp_data = data_processing(data)
    normalized_temp = data_scaling(temp_data)
    series = series_to_supervised(normalized_temp, window=24)
    X_train,Y_train,X_valid,Y_valid = train_test_split(series)
    X_train_reshaped, X_valid_reshaped = reshaping(X_train,X_valid)

    # Normalized predictions:
    lstm_train_pred = model.predict(X_train_reshaped)
    lstm_valid_pred = model.predict(X_valid_reshaped)

    print(lstm_train_pred)
    print(lstm_valid_pred)

    return jsonify({'true_train': Y_train,
                    'pred_train':lstm_train_pred,
                    'true_val': Y_valid,
                    'pred_val': lstm_valid_pred})