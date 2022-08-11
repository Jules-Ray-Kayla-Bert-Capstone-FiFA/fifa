import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def create_baseline(y_train, y_validate, target):
    
    pd.options.display.float_format = '{:20.2f}'.format 
    '''
    Take in y_train and y_validate dataframe and target variable(wage_euro). 
    Calculate the mean and median of the target variable and print the result side by side comparison
    Select the one that has the lowest RMSE
    Append into a dataframe called metric_df
    '''
    #wage_euro_value mean
    wage_euro_pred_mean = y_train[target].mean()
    y_train['wage_euro_pred_mean'] = wage_euro_pred_mean
    y_validate['wage_euro_pred_mean'] = wage_euro_pred_mean

    #wage_euro_value_median
    wage_euro_pred_median = y_train[target].median()
    y_train['wage_euro_pred_median'] = wage_euro_pred_median
    y_validate['wage_euro_pred_median'] = wage_euro_pred_median


    #RMSE of wage_euro_value_pred_mean
    rmse_mean_train = mean_squared_error(y_train[target], y_train.wage_euro_pred_mean)**(1/2)
    rmse_mean_validate = mean_squared_error(y_validate[target], y_validate.wage_euro_pred_mean)**(1/2)


    #RMSE of wage_euro_value_pred_median
    rmse_median_train = mean_squared_error(y_train[target], y_train.wage_euro_pred_median)**(1/2)
    rmse_median_validate = mean_squared_error(y_validate[target], y_validate.wage_euro_pred_median)**(1/2)

    #R2 score for the baseline
    r2_baseline = r2_score(y_validate[target], y_validate.wage_euro_pred_mean)

    #Append rmse validate and r2 score into a dataframe
    metric_df = pd.DataFrame(data=[{
    'model': 'Mean Baseline',
    'rmse_train': rmse_mean_train,
    'rmse_validate': rmse_mean_validate,
    'r^2_value': r2_baseline}])

    return  metric_df, rmse_mean_train, rmse_mean_validate, rmse_median_train, rmse_median_validate, r2_baseline

def create_model(model, X_train, X_validate, y_train, y_validate, target):
    
    '''
    Function takes in X_train df and target df (wage_eur), and
    type of model (LinearRegression, LassoLars, TweedieRegressor) and
    calculates the mean squared error and the r2 score
    Finally, returns mean squared error and the r2 score
    '''
    #fit the model to our training data, specify column since it is a dataframe
    model.fit(X_train, y_train[target])

    #predict train
    y_train['wage_euro_pred_lm'] = model.predict(X_train)
    y_train['wage_euro_pred_lars'] = model.predict(X_train)
    y_train['wage_euro_pred_glm'] = model.predict(X_train)

    #evaluate the RMSE for train
    rmse_train = mean_squared_error(y_train[target], y_train.wage_euro_pred_lm)**(1/2)
    rmse_train = mean_squared_error(y_train[target], y_train.wage_euro_pred_lars)**(1/2)
    rmse_train = mean_squared_error(y_train[target], y_train.wage_euro_pred_glm)**(1/2)
    
    #predict validate
    y_validate['wage_euro_pred_lm'] = model.predict(X_validate)
    y_validate['wage_euro_pred_lars'] = model.predict(X_validate)
    y_validate['wage_euro_pred_glm'] = model.predict(X_validate)
   
    #evaluate the RMSE for validate
    rmse_validate = mean_squared_error(y_validate[target], y_validate.wage_euro_pred_lm)**(1/2)
    rmse_validate = mean_squared_error(y_validate[target], y_validate.wage_euro_pred_lars)**(1/2)
    rmse_validate = mean_squared_error(y_validate[target], y_validate.wage_euro_pred_glm)**(1/2)


    #r2 score for model
    r2_model_score = r2_score(y_validate[target], y_validate.wage_euro_pred_lm)
    r2_model_score = r2_score(y_validate[target], y_validate.wage_euro_pred_lars)
    r2_model_score = r2_score(y_validate[target], y_validate.wage_euro_pred_glm)

    # lm residuals
    y_train['lm_residuals'] = y_train['wage_euro_pred_lm'] - y_train['wage_euro']
    y_validate['lm_residuals'] = y_validate['wage_euro_pred_lm'] - y_validate['wage_euro']

    #lars residuals
    y_train['lars_residuals'] = y_train['wage_euro_pred_lars'] - y_train['wage_euro']
    y_validate['lars_residuals'] = y_validate['wage_euro_pred_lars'] - y_validate['wage_euro']

    #glm residuals
    y_train['glm_residuals'] = y_train['wage_euro_pred_glm'] - y_train['wage_euro']
    y_validate['glm_residuals'] = y_validate['wage_euro_pred_glm'] - y_validate['wage_euro']


    return y_train, X_train, print(f'RMSE Train: ${rmse_train:.2f}'), print(f'RMSE Validate: ${rmse_validate:.2f}'), print(f'R\u00b2: {r2_model_score:.3f}')

def RMSE_function(X_train,X_validate, y_train, y_validate,target):
    #create baseline
    metric_df, rmse_mean_train, rmse_mean_validate, rmse_median_train, rmse_median_validate, r2_baseline = create_baseline(y_train, y_validate, target)
    #Linear Regression model
    rmse_lm_train, rmse_lm_validate, r2_lm_value = create_model(LinearRegression(normalize = True), X_train,\
                                                                    X_validate, y_train, y_validate, 'wage_eur')
    #Lasso + Lars model
    rmse_lars_train, rmse_lars_validate, r2_lars_value = create_model(LassoLars(alpha = 1.0), X_train,\
                                                                X_validate, y_train, y_validate, 'wage_eur')
    #Tweedie Regressor model
    rmse_glm_train, rmse_glm_validate, r2_glm_value = create_model(TweedieRegressor(power = 1, alpha = 0.00),\
                                                                X_train, X_validate, y_train, y_validate, 'wage_eur')
    print(metric_df)
