import pandas as pd

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error

class TimePredictionModel: #Class containing all our model.
    def __init__(self, model=LinearRegression(), transform=None, reverse_transform=None, round_=False):
        self.model = model
        self.transform = transform #Transform function, could be log for example
        self.reverse_transform = reverse_transform #Reverse Transform, Both Transform and Reverse Transform MUST be defined to transform the data.
        self.dictvectorizer = DictVectorizer()
        
        self.model_trained = False
        self.round = round_ #Whether to round the result or not. I've found that the data is highly skewed towards whole minute
    def fit(self, train_dict, y_train):
        assert  not self.model_trained, "Model already trained"
            
        X_train = self.dictvectorizer.fit_transform(train_dict)
        
        if self.transform == None or self.reverse_transform == None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, self.transform(y_train))
        self.model_trained = True
        
    def predict(self, test_dict):
        assert self.model_trained, "Model Not yet trained, train with model.fit"
        X_test = self.dictvectorizer.transform(test_dict)
        
        if self.transform == None or self.reverse_transform == None:
            if self.round:
                return np.round(self.model.predict(X_test))
            else:
                return self.model.predict(X_test)
        else:
            if self.round:
                return np.round(self.reverse_transform(self.model.fit(X_test)))
            else:
                return self.reverse_transform(self.model.fit(X_test))
        
    def transform(self, test_dict):
        assert self.model_trained, "Model Not yet trained, train with model.fit"
        
        return self.dictvectorizer.transform(test_dict)
        