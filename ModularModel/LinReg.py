import pandas as pd

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

from preprocess_simple_green import preprocess, categorical, numerical, target

def vectorize(train_path, val_path):
    
    df_train = preprocess(train_path)
    df_val = preprocess(val_path)
    
    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
a
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

# Modelling
def train(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_val)

    print(mean_squared_error(y_val, y_pred, squared=False))

    return lr

train_path = "../data/green_tripdata_2021-01.parquet"
val_path = "../data/green_tripdata_2021-02.parquet"


X_train, X_val, y_train, y_val, dv = vectorize(train_path, val_path) 
train(X_train, y_train)

def save(lr, dv):
    with open('models/lin_reg.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)

    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

