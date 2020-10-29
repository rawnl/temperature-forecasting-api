#from datetime import datetime
import datetime
import pandas as pd
import tensorflow

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import joblib

from app import create_app, db

app = create_app()

new_model = tensorflow.keras.models.load_model("app/models/new_simple_model.h5")

temp_scaler = joblib.load('app/models/original_scaler.save')

'''
    data = pd.read_csv('app/data/climate_hour.csv',index_col = 0, header=0)#, encoding= 'unicode_escape' #,skiprows=5,nrows=200) #parse_dates=['Date Time'],
    #data = data.sort_values(['Date Time'])
    
    temp_data = data_processing(data)
    
    normalized_temp = data_scaling(temp_data)
    
    series = series_to_supervised(normalized_temp, window=144)
    
    labels_col = 'T (degC)(t)'
    labels = series[labels_col]
    series = series.drop(['T (degC)(t)'], axis=1)
    
    X_train = series[:160] 
    X_valid = series[160:]
    Y_train = labels[:160]
    Y_valid = labels[160:]
    '''

def data_processing(data):
    temp_data = data["T (degC)"]
    temp_data = pd.DataFrame({"Date Time": data.index, "T (degC)":temp_data.values})
    temp_data = temp_data.set_index(["Date Time"])
    return temp_data

def data_scaling(data):
    normalized_temp = temp_scaler.transform(data)
    normalized_temp = pd.DataFrame(normalized_temp, columns=['T (degC)'])
    normalized_temp.index = data.index
    return normalized_temp

def data_transforming(prediction):
    sample = temp_scaler.inverse_transform(prediction) #Y_valid.values.reshape(-1,1)
    return sample

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
    series = series.drop(['T (degC)(t-144)'], axis=1)
    
    """
    X_train = series['2009-01-02 01:00:00':'01.01.2015 00:00:00']
    X_valid = series['01.01.2015 00:00:00':'2017-01-01 00:00:00']
    Y_train = labels['2009-01-02 01:00:00':'01.01.2015 00:00:00']
    Y_valid = labels['01.01.2015 00:00:00':'2017-01-01 00:00:00']
    return X_train,Y_train,X_valid,Y_valid
    """
    return series
    
def reshaping(series):
    # X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    print("series[0]:{}".format(series.shape[0]))
    print("series[1]:{}".format(series.shape[1]))
    X = series.values.reshape((series.shape[0], series.shape[1], 1)) #series.shape[0], series.shape[1], 1
    return X


# scheduler job:
def predictions_generator():
    print('job started.')
     
    #get data
    data = pd.read_csv('app/data/climate_hour.csv',index_col = 0, header=0, nrows=145)
    
    #data preprocessing
    temp_data = data_processing(data)
    
    #normalize data
    normalized_temp = data_scaling(temp_data) #temp_data
    '''print("normalized")
    print(normalized_temp)'''

    series = series_to_supervised(normalized_temp,144)
    print('series = {}'.format(series))

    series = train_test_split(series)
    
    series = reshaping(series)
    
    """X = []
    for e in normalized_temp:
        print(e)
        el = [e]
        X.append(el)
    X = reshaping(X)
    print(X)
    """
    
    # make predictions 
    pred = new_model.predict(series)  
    print("pred :{}".format(pred))  
    
    converted_pred = data_transforming(pred)
    print('converted : {}'.format(converted_pred))

    #row = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    row = datetime.datetime.now().replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    row = str('{},{}'.format(row,converted_pred))
    print(row)
    #row = row.replace(second=0, microsecond=0)

    # save to db/csv 
    with open('generated_forecasts.csv','a') as fd:
        fd.write(str(row))

    # save to db 
    '''
    date_hour = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)# + datetime.timedelta(hours=1)
    forecast = HourlyForecasting(date_hour=date_hour,temperature='21')
    db.session.add(forecast)
    db.session.commit()
    '''
   
    print('job ended.')

#scheduled script :
def start_scheduler():
    # define a background schedule
    # Attention: you cannot use a blocking scheduler here as that will block the script from proceeding.
    scheduler = BackgroundScheduler({'apscheduler.timezone': 'UTC'})

    # define your job trigger
    trigger = CronTrigger(hour='*', minute='03')

    # add your job
    scheduler.add_job(func=predictions_generator, trigger=trigger, max_instances=1)

    # start the scheduler
    scheduler.start()
    

'''
def start_scheduler():
    # define a background schedule
    # Attention: you cannot use a blocking scheduler here as that will block the script from proceeding.
    scheduler = BackgroundScheduler()

    # define your job trigger
    #hourse_keeping_trigger = CronTrigger(hour='12', minute='30')

    # add your job
    #scheduler.add_job(func=run_housekeeping, trigger=hourse_keeping_trigger)

    # start the scheduler
    scheduler.start()
'''

# run server
if __name__ == '__main__':
    start_scheduler()
    app.run(debug=True, threaded=False, use_reloader=False)

