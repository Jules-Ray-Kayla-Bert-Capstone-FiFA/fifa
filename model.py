import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor

from sklearn.metrics import classification_report



def score_models(X_train, y_train, X_val, y_val):
    '''
    Score multiple models on train and val datasets.
    Print classification reports to decide on a model to test.
    Return each trained model, so we can choose one to test.
    models = lr_model, dt_model, rf_model, kn_model.
    '''
    lr_model = LinearRegression(normalize=True)
    ll_model = LassoLars(alpha=1.0)
    tr_model = TweedieRegressor(power=1, alpha=0)
    models = [lr_model, ll_model, tr_model]
    for model in models:
        model.fit(X_train, y_train)
        actual_train = y_train
        predicted_train = model.predict(X_train)
        actual_val = y_val
        predicted_val = model.predict(X_val)
        print(model)
        print('')
        print('train score: ')
        print(classification_report(actual_train, predicted_train))
        print('val score: ')
        print(classification_report(actual_val, predicted_val))
        print('________________________')
        print('')
    return lr_model, ll_model, tr_model