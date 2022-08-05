import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm
import math
from math import sqrt

def modeling_function(X_train, y_train, X_test, y_test):
    predictions=pd.DataFrame({"actual":y_train.wage_eur}).reset_index(drop=True)
    predictions_test=pd.DataFrame({"actual":y_test.wage_eur}).reset_index(drop=True)

    lm1=LinearRegression()
    lm1.fit(X_train,y_train)
    lm1_predictions=lm1.predict(X_train)
    predictions["lm1"]=lm1_predictions

    #model test
    lm1_test=LinearRegression()
    lm1_test.fit(X_test,y_test)
    lm1_test_predictions=lm1_test.predict(X_test)
    predictions_test["lm1_test"]=lm1_test_predictions

    #model baseline
    predictions["lm_baseline"] = y_train.wage_eur.mean()
    predictions_test["lm_baseline_test"] = y_test.wage_eur.mean()

    return predictions, predictions_test


def plot_residuals(actual, predicted, feature):
    """
    Returns the scatterplot of actural y in horizontal axis and residuals in vertical axis
    Parameters: actural y(df.se), predicted y(df.se), feature(str)
    Prerequisite: call function evaluate_slr
    """
    plt.figure(figsize=(20,10))
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title(f'Actual vs Residual on {feature}')
    return plt.gca()

def validate_rmse(y_train, y_validate):
    '''
    Computes the root mean squared error of the baseline model (the mean of y_train) 
    when compared to the y_validate targets
    '''
    baseline_validate_residual = y_validate - y_train.mean()
    baseline_validate_sse = (baseline_validate_residual**2).sum()
    baseline_validate_mse = baseline_validate_sse/y_validate.size
    v_rmse = math.sqrt(baseline_validate_mse)
    return v_rmse

def test_rmse(y_train, y_test):
    '''
    Computes the root mean squared error of the baseline model (the mean of y_train)
    when compared to the y_test targets
    '''
    baseline_test_residual = y_test - y_train.mean()
    baseline_test_sse = (baseline_test_residual**2).sum()
    baseline_test_mse = baseline_test_sse/y_test.size
    t_rmse = math.sqrt(baseline_test_mse)
    return t_rmse 


