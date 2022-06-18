import pandas as pd

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

def read_dataframe(filepath):
    df = pd.read_parquet(filepath)
    
    print(df.columns.tolist())

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def preprocess(filepath):
    df = read_dataframe(filepath)

    print(len(df))

    df['PU_DO'] = df['PUlocationID'] + '_' + df['DOlocationID']

    return df

categorical = ['PU_DO']
numerical = ['trip_distance']