import pandas as pd
import numpy as np

import holidays

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

def preprocess(filepath, floor=True):
    data = pd.read_parquet(filepath)
    data['time'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
    if floor: #Floor or not Floor, the data is skewed. Check next cell.
        data['duration'] = np.floor(data.time.dt.total_seconds() / 60 )
    else:
        data['duration'] = data.time.dt.total_seconds() / 60 
    
    data = data.loc[data.duration >= 4].loc[data.duration <= 60] #I've used 4 minutes as the cutoff as it appears the distribution
                                                                         #is Highly erroneous for any time less than that. Could also bundle all times
                                                                         #less than 5 minutes under one label.

    
    us_holidays = holidays.US()
    data['date'] = data.lpep_dropoff_datetime.dt.date
    data['holiday'] = data.date.apply(lambda date: us_holidays.get(date)) #Get holidays for each date.
    
    data['dayofweek'] = data.lpep_dropoff_datetime.dt.dayofweek
    data['departure_hour'] = data.lpep_dropoff_datetime.dt.hour
    
    data.drop(data.columns.difference(['duration','trip_distance', 'PULocationID','DOLocationID', 'holiday', 'dayofweek', 'departure_hour']), axis=1, inplace=True)
    
    data['holiday_bin'] = data.holiday != 'None' #Treat all holidays the same. Necessary as our datasets are within 2 different months.
                                                 #A better analysis would involve the entire year.
    data.drop('holiday', axis=1, inplace=True)
    
    categorical = ['PULocationID','DOLocationID', 'holiday_bin', 'dayofweek', 'departure_hour']
    numerical =['trip_distance']

    data[categorical] = data[categorical].astype('str')
    data['PU_DO'] = data.PULocationID + data.DOLocationID
    
    categorical = ['PU_DO', 'PULocationID', 'DOLocationID', 'holiday_bin', 'dayofweek', 'departure_hour']
    
    return data

categorical = ['PU_DO', 'PULocationID', 'DOLocationID', 'holiday_bin', 'dayofweek', 'departure_hour']
numerical =['trip_distance']
target= 'duration'