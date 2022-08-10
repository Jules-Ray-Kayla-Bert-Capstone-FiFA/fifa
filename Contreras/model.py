import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale = ['overall', 'potential', 'age', 'height_cm', 'weight_kg', 'club_team_id', 'nationality_id', 
        'weak_foot', 'skill_moves', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical',
        'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'skill_dribbling', 
        'curve', 'fk_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility',
        'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
        'interceptions', 'positioning', 'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
        'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes', 'gk_speed', 'year', 
        'year_joined', 'seniority', 'work_rate_encoded', 'preferred_foot_encoded',
        'age_bins_encoded', 'weight_bins_encoded', 'body_type_encoded', 'league_encoded'],
               return_scaler = False):

    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    #create variables from train, validate, and test
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #create scaler
    scaler = RobustScaler()
    #fit only to train
    scaler.fit(train[columns_to_scale])
    
    #scale on numerical columns
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns = train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns = validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns = test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


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
    

    return print(f'RMSE Train: ${rmse_train:.2f}'), print(f'RMSE Validate: ${rmse_validate:.2f}'), print(f'R\u00b2: {r2_model_score:.3f}')