MLFLOW_LOG = False


import pandas as pd

import pickle

from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import mean_squared_error

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import xgboost as xgb

from preprocessors.preprocess_complex_green import preprocess, categorical, numerical, target

if MLFLOW_LOG:
    import mlflow
    mlflow.set_tracking_uri("sqlite:////home/qfl3x/mlflow/mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

def vectorize(train_path, val_path): #Vectorizer
    
    df_train = preprocess(train_path)
    df_val = preprocess(val_path)
    
    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

train_path = "../data/green_tripdata_2021-01.parquet"
val_path = "../data/green_tripdata_2021-02.parquet"

X_train, X_val, y_train, y_val, dv = vectorize(train_path, val_path) 

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)
if MLFLOW_LOG:
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("preprocessor", "complex")
        mlflow.log_params(params)
        booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
                )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}
else:
    def objective(params):
        booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
                )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        return {'loss': rmse, 'status': STATUS_OK}

search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
