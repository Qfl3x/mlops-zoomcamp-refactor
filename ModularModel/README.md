Proprocessor file convention:

`preprocess\_[Complex/Simple]\_[Data type: green/fhv].py`

\[Complex/Simple]: Type of preprocessor; Simple: Course preprocessor. Complex: Preprocessor with weekdays/weekends encoding and holidays. Requires `holidays` package.

\[Data type]: The type of data we wish to train/predict.

`LinReg.py`: Linear Regression model.

`XGBoost_Search.py`: `hyperopt` paramater search for XGBoost. Has `MLFLOW_LOG` flag, which dicates whether or not the results of each run are logged in MLflow.
