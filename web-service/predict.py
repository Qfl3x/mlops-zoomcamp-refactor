import pandas as pd

import pickle

import holidays

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

from model import TimePredictionModel


def process_data(data, floor=True, predict=True):
    data = data.copy()
    if not predict:
        data['time'] = data.lpep_dropoff_datetime - data.lpep_pickup_datetime
        if floor: #Floor or not Floor, the data is skewed. Check next cell.
            data['time_minutes'] = np.floor(data.time.dt.total_seconds() / 60 )
        else:
            data['time_minutes'] = data.time.dt.total_seconds() / 60 

        data = data.loc[data.time_minutes >= 4].loc[data.time_minutes <= 60] #I've used 4 minutes as the cutoff as it appears the distribution
                                                                             #is Highly erroneous for any time less than that. Could also bundle all times
                                                                             #less than 5 minutes under one label.

    
    us_holidays = holidays.US()
    data['date'] = data.lpep_pickup_datetime.dt.date
    data['holiday'] = data.date.apply(lambda date: us_holidays.get(date)) #Get holidays for each date.
    
    data['dayofweek'] = data.lpep_pickup_datetime.dt.dayofweek
    data['departure_hour'] = data.lpep_pickup_datetime.dt.hour
    
    data.drop(data.columns.difference(['time_minutes','trip_distance', 'PULocationID','DOLocationID', 'holiday', 'dayofweek', 'departure_hour']), axis=1, inplace=True)
    
    data['holiday_bin'] = data.holiday != 'None' #Treat all holidays the same. Necessary as our datasets are within 2 different months.
                                                 #A better analysis would involve the entire year.
    data.drop('holiday', axis=1, inplace=True)
    
    categorical = ['PULocationID','DOLocationID', 'holiday_bin', 'dayofweek', 'departure_hour']
    numerical =['trip_distance']

    data[categorical] = data[categorical].astype('str')
    data['PU_DO'] = data.PULocationID + data.DOLocationID
    
    categorical = ['PU_DO', 'PULocationID', 'DOLocationID', 'holiday_bin', 'dayofweek', 'departure_hour']
    data_dict = data[categorical + numerical].to_dict(orient='records')
    
    if predict:
        return data_dict
    else:
        return data, data_dict, data['time_minutes']

with open("linreg.bin", 'rb') as f_in:
    Model = pickle.load(f_in)
    
    
def predict(features):
    features = pd.DataFrame.from_dict(features, orient='columns')
    X_dict = process_data(features)
    return Model.predict(X_dict)


from flask import Flask, request, jsonify

app = Flask('duration_predict')

@app.route('/endpoint_predict',methods=['POST','GET'])

def endpoint_predict():
    ride = request.get_json()
    
    ride['lpep_pickup_datetime'] = pd.to_datetime(ride['lpep_pickup_datetime'])
    features = {key: [ride[key]] for key in ride.keys()}
    #return features
    pred = predict(features)
    return_dict = {'duration': pred[0]}
    return jsonify(return_dict)



    
    
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9696)