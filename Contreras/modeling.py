import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression 

#statistical tests
from scipy import stats
from scipy.stats import pearsonr, spearmanr

from math import sqrt
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup
import prepare
import plotly.express as px

#imports to show interactive visuals on github
import plotly.io as pio
pio.renderers

import modeling
#import model
import math
from math import sqrt

#################################################################
def handle_outliers(df, cols, k):
    """this will eliminate most outliers, use a 1.5 k value if unsure because it is the most common, make sure to define cols value as the features
    you want the outliers to be handled. this should be done before running the function and outiside of it"""

    
    # Create placeholder dictionary for each columns bounds
    bounds_dict = {}
   
    for col in cols:
        # get necessary iqr values
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr

        #store values in a dictionary referencable by the column name
        #and specific bound
        bounds_dict[col] = {}
        bounds_dict[col]['upper_bound'] = upper_bound
        bounds_dict[col]['lower_bound'] = lower_bound

    for col in cols:
        #retrieve bounds
        col_upper_bound = bounds_dict[col]['upper_bound']
        col_lower_bound = bounds_dict[col]['lower_bound']

        #remove rows with an outlier in that column
    df = df[(df[col] < col_upper_bound) & (df[col] > col_lower_bound)]
        
    return df
#################################################################
def create_baseline(y_train, y_validate, target):
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict wage_eur_pred_mean
    wage_eur_pred_mean = y_train['wage_eur'].mean()
    y_train['wage_eur_pred_mean'] = wage_eur_pred_mean
    y_validate['wage_eur_pred_mean'] = wage_eur_pred_mean
    
    # 2. compute wage_eur_pred_median
    wage_eur_pred_median = y_train['wage_eur'].median()
    y_train['wage_eur_pred_median'] = wage_eur_pred_median
    y_validate['wage_eur_pred_median'] = wage_eur_pred_median
    
    # 3. RMSE of wage_eur_pred_mean
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_mean)**(1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    # 4. RMSE of wage_eur_pred_median
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_median)**(1/2)
    
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))


################################################################
def linear_model(y_train,y_validate,X_train,X_validate):
    
    '''
    Function takes in X_train df and target df (wage_eur), and
    type of model (LinearRegression, LassoLars, TweedieRegressor) and
    calculates the mean squared error and the r2 score
    Finally, returns mean squared error and the r2 score
    '''
    # create the model object
    lm = LinearRegression(normalize=True)
    
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.wage_eur)
    
    # predict train
    y_train['wage_eur_pred_lm'] = lm.predict(X_train)
    
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_lm)**(1/2)
    
    # predict validate
    y_validate['wage_eur_pred_lm'] = lm.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate_lm)
################################################################    
def lassolars_model(y_train,y_validate,X_train,X_validate):
    # create the model object
    lars = LassoLars(alpha=1.0)
    
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.wage_eur)
    
    # predict train
    y_train['wage_eur_pred_lars'] = lars.predict(X_train)
    
    # evaluate: rmse
    rmse_train_lars = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_lars)**(1/2)
    
    # predict validate
    y_validate['wage_eur_pred_lars'] = lars.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate_lars = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_lars)**(1/2)
    
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train_lars, 
          "\nValidation/Out-of-Sample: ", rmse_validate_lars)
################################################################# 
def tweedie_model(y_train,y_validate,X_train,X_validate):
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.wage_eur)
    
    # predict train
    y_train['wage_eur_pred_glm'] = glm.predict(X_train)
    
    # evaluate: rmse
    rmse_train_glm = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_glm)**(1/2)
    
    # predict validate
    y_validate['wage_eur_pred_glm'] = glm.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate_glm = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_glm)**(1/2)
    
    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ",rmse_train_glm, 
      "\nValidation/Out-of-Sample: ", rmse_validate_glm)
################################################################   
def metrics(df,y_train,y_validate,X_train,X_validate):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict wage_eur_pred_mean
    wage_eur_pred_mean = y_train['wage_eur'].mean()
    y_train['wage_eur_pred_mean'] = wage_eur_pred_mean
    y_validate['wage_eur_pred_mean'] = wage_eur_pred_mean
    
    # 2. compute wage_eur_pred_median
    wage_eur_pred_median = y_train['wage_eur'].median()
    y_train['wage_eur_pred_median'] = wage_eur_pred_median
    y_validate['wage_eur_pred_median'] = wage_eur_pred_median
    
    # 3. RMSE of wage_eur_pred_mean
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_mean)**(1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    # 4. RMSE of wage_eur_pred_median
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_median)**(1/2)
    # create the model object
    lm = LinearRegression(normalize=True)
    
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.wage_eur)
    
    # predict train
    y_train['wage_eur_pred_lm'] = lm.predict(X_train)
    
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_lm)**(1/2)
    
    # predict validate
    y_validate['wage_eur_pred_lm'] = lm.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate_lm = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_lm)**(1/2)
    ##############################################################
    # create the model object
    lars = LassoLars(alpha=1.0)
    
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.wage_eur)
    
    # predict train
    y_train['wage_eur_pred_lars'] = lars.predict(X_train)
    
    # evaluate: rmse
    rmse_train_lars = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_lars)**(1/2)
    
    # predict validate
    y_validate['wage_eur_pred_lars'] = lars.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate_lars = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_lars)**(1/2)
    #########################################################
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.wage_eur)
    
    # predict train
    y_train['wage_eur_pred_glm'] = glm.predict(X_train)
    
    # evaluate: rmse
    rmse_train_glm = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_glm)**(1/2)
    
    # predict validate
    y_validate['wage_eur_pred_glm'] = glm.predict(X_validate)
    
    # evaluate: rmse
    rmse_validate_glm = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_glm)**(1/2)
    ###########################################################
    # sklearn.metrics.explained_variance_score
    from sklearn.metrics import explained_variance_score
    evs = explained_variance_score(df.wage_eur, df.yhat)
    print('Explained Variance = ', round(evs,3))
    rmse_validate = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_median)**(1/2)
    metric_df = pd.DataFrame(data=[{
    'model': 'mean_baseline', 
    'RMSE_validate': rmse_validate,
        'r^2_validate': explained_variance_score(y_validate.wage_eur, y_validate.wage_eur_pred_mean)}])
    metric_df = metric_df.append({
    'model': 'OLS Regressor', 
    'RMSE_validate': rmse_validate_lm,
    'r^2_validate': explained_variance_score(y_validate.wage_eur, y_validate.wage_eur_pred_lm)}, ignore_index=True)
    metric_df = metric_df.append({
    'model': 'Lasso alpha 1', 
    'RMSE_validate': rmse_validate_lars,
    'r^2_validate': explained_variance_score(y_validate.wage_eur, y_validate.wage_eur_pred_lars)}, ignore_index=True)
    metric_df = metric_df.append({
    'model': 'GLM (tweedie)', 
    'RMSE_validate': rmse_validate_glm,
    'r^2_validate': explained_variance_score(y_validate.wage_eur, y_validate.wage_eur_pred_glm)}, ignore_index=True)
    return metric_df 
#################################################################
def actual_vs_predicted(train,y_train,y_validate,X_train,X_validate,y_test):
    from sklearn.linear_model import LinearRegression
    #residuals
    y_train['lm_residuals'] = y_train.wage_eur_pred_lm - y_train['wage_eur']
    y_validate['lm_residuals'] = y_validate.wage_eur_pred_lm - y_validate['wage_eur']
    #residuals
    y_train['lars_residuals'] = y_train['wage_eur_pred_lars'] - y_train['wage_eur']
    y_validate['lars_residuals'] = y_validate['wage_eur_pred_lars'] - y_validate['wage_eur']
    #residuals
    y_train['glm_residuals'] = y_train['wage_eur_pred_glm'] - y_train['wage_eur']
    y_validate['glm_residuals'] = y_validate['wage_eur_pred_glm'] - y_validate['wage_eur']
    y_test = pd.DataFrame(y_test)
    y = pd.DataFrame(y_train.wage_eur)
    X = pd.DataFrame(X_train)
    # assuming X and y are already defined
    m = LinearRegression().fit(X, y)
    train['yhat'] = m.predict(X)
    df = pd.DataFrame(train[['international_reputation','overall','wage_eur','yhat']])
    df['baseline'] = y.wage_eur.mean()
    # turning baseline to int from float
    df.baseline = df.baseline.astype(int)
    # residual = actual - predicted
    df['residual'] = df.wage_eur - df.yhat
    df['baseline_residual'] = df.wage_eur - df.baseline
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.wage_eur, color='blue', alpha=.5, label="Actual Final wage_eur")
    plt.hist(y_validate.wage_eur_pred_lm, color='red', alpha=.5, label="Model: LinearRegression")
    plt.hist(y_validate.wage_eur_pred_glm, color='yellow', alpha=0.5, label="Model: TweedieRegressor")
    plt.hist(y_validate.wage_eur_pred_lars, color='green', alpha=.5, label="Model: Lasso Lars")
    plt.xlabel("Final wage (eur)")
    plt.ylabel("predicted wage (eur")
    plt.title("Comparing the Distribution of Actual wage_eur weekly to Distributions of Predicted wage_eur for the Top Models")
    plt.legend()
    plt.show()
################################################################    
def test(y_train,y_validate,X_train,X_validate,y_test,X_test):
    
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # 3. RMSE of wage_eur_pred_mean
    rmse_train = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_mean)**(1/2)

    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.wage_eur)

    # predict train
    y_train['wage_eur_pred_glm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train_glm = mean_squared_error(y_train.wage_eur, y_train.wage_eur_pred_glm)**(1/2)

    # predict validate
    y_validate['wage_eur_pred_glm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate_glm = mean_squared_error(y_validate.wage_eur, y_validate.wage_eur_pred_glm)**(1/2)
    y_test = pd.DataFrame(y_test)
    # ###########################################################

    y_test = pd.DataFrame(y_test)

    #predict on test
    y_test['wage_eur_pred_glm'] = glm.predict(X_test)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.wage_eur, y_test.wage_eur_pred_glm)**(1/2)
    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate_glm)
#################################################################    
def overall_modeling(df):
    print("================================               OVERALL                 =====================================")
    print("splitting data")
    train, validate, test = prepare.split(df)
    print("=========================================== Model for Overall Data =========================================")
    # identifying mvp columns
    mvp = ['international_reputation','overall','reactions','potential']
    print("The features that we will use for the model:", mvp)
    X_train = train[mvp]
    y_train = train[['wage_eur']]

    X_validate = validate[mvp]
    y_validate = validate[['wage_eur']]

    X_test = test[mvp]
    y_test = test[['wage_eur']]
    modeling.create_baseline(y_train, y_validate,' wage_eur')
    modeling.tweedie_model(y_train,y_validate,X_train,X_validate)
    print("Lets take a look at the numbers:")
    print("train")
    print(y_train)
    print("validate")
    print(y_validate)
    print("============================================= Test Data ==================================================")
    modeling.test(y_train,y_validate,X_train,X_validate,y_test,X_test)
    print(y_test)
    fig = px.histogram(y_test.sample(n=1800, replace=False, random_state=123).sort_index())
    fig.show()
#################################################################
def forwards_modeling(df):
    goalkeeper_df, forward_df, midfielder_df, defender_df = prepare.acquire_players_by_position(df)
    print("================================               FORWARDS                  =====================================")
    df = forward_df
    print("splitting data")
    train, validate, test = prepare.split(df)
    print("=======================================        Model Data            =========================================")
    # identifying mvp columns
    mvp = ['overall','shooting','ball_control']
    print("The features that we will use for the model:", mvp)
    X_train = train[mvp]
    y_train = train[['wage_eur']]
    
    X_validate = validate[mvp]
    y_validate = validate[['wage_eur']]
    
    X_test = test[mvp]
    y_test = test[['wage_eur']]
    modeling.create_baseline(y_train, y_validate,' wage_eur')
    modeling.tweedie_model(y_train,y_validate,X_train,X_validate)
    print("Lets take a look at the numbers:")
    print("train")
    print(y_train)
    print("validate")
    print(y_validate)
    print("============================================= Test Data ====================================================")
    modeling.test(y_train,y_validate,X_train,X_validate,y_test,X_test)
    print(y_test)
    fig = px.histogram(y_test.sample(n=1800, replace=False, random_state=123).sort_index())
    fig.show()
#################################################################
def midfielder_modeling(df):
    goalkeeper_df, forward_df, midfielder_df, defender_df = prepare.acquire_players_by_position(df)
    print("================================               MIDFIELDERS               =====================================")
    df = midfielder_df
    print("splitting data")
    train, validate, test = prepare.split(df)
    print("=======================================        Model Data            =========================================")
    # identifying mvp columns
    mvp = ['international_reputation','overall','passing']
    print("The features that we will use for the model:", mvp)
    X_train = train[mvp]
    y_train = train[['wage_eur']]

    X_validate = validate[mvp]
    y_validate = validate[['wage_eur']]

    X_test = test[mvp]
    y_test = test[['wage_eur']]
    modeling.create_baseline(y_train, y_validate,' wage_eur')
    modeling.tweedie_model(y_train,y_validate,X_train,X_validate)
    print("Lets take a look at the numbers:")
    print("train")
    print(y_train)
    print("validate")
    print(y_validate)
    print("============================================= Test Data ====================================================")
    modeling.test(y_train,y_validate,X_train,X_validate,y_test,X_test)
    print(y_test)
    fig = px.histogram(y_test.sample(n=1800, replace=False, random_state=123).sort_index())
    fig.show()
#################################################################    
def defender_modeling(df):
    goalkeeper_df, forward_df, midfielder_df, defender_df = prepare.acquire_players_by_position(df)
    print("================================               DEFENDERS                 =====================================")
    df = defender_df
    print("splitting data")
    train, validate, test = prepare.split(df)
    print("=======================================        Model Data            =========================================")
    # identifying mvp columns
    mvp = ['overall','defending']
    print("The features that we will use for the model:", mvp)
    X_train = train[mvp]
    y_train = train[['wage_eur']]

    X_validate = validate[mvp]
    y_validate = validate[['wage_eur']]

    X_test = test[mvp]
    y_test = test[['wage_eur']]
    modeling.create_baseline(y_train, y_validate,' wage_eur')
    modeling.tweedie_model(y_train,y_validate,X_train,X_validate)
    print("Lets take a look at the numbers:")
    print("train")
    print(y_train)
    print("validate")
    print(y_validate)
    print("============================================= Test Data ====================================================")
    modeling.test(y_train,y_validate,X_train,X_validate,y_test,X_test)
    print(y_test)
    fig = px.histogram(y_test.sample(n=1800, replace=False, random_state=123).sort_index())
    fig.show()
#################################################################
def goalkeeper_modeling(df):
    goalkeeper_df, forward_df, midfielder_df, defender_df = prepare.acquire_players_by_position(df)
    print("================================              GOAL KEEPER                =====================================")
    df = goalkeeper_df
    print("splitting data")
    train, validate, test = prepare.split(df)
    print("=======================================        Model Data            =========================================")
    # identifying mvp columns
    mvp = ['overall','gk_reflexes']
    print("The features that we will use for the model:", mvp)
    X_train = train[mvp]
    y_train = train[['wage_eur']]

    X_validate = validate[mvp]
    y_validate = validate[['wage_eur']]

    X_test = test[mvp]
    y_test = test[['wage_eur']]
    modeling.create_baseline(y_train, y_validate,' wage_eur')
    modeling.tweedie_model(y_train,y_validate,X_train,X_validate)
    print("Lets take a look at the numbers:")
    print("train")
    print(y_train)
    print("validate")
    print(y_validate)
    print("============================================= Test Data ====================================================")
    modeling.test(y_train,y_validate,X_train,X_validate,y_test,X_test)
    print(y_test)
    fig = px.histogram(y_test.sample(n=1000, replace=False, random_state=123).sort_index())
    fig.show()