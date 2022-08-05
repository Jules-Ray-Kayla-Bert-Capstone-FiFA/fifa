#########################################################- IMPORTS -###########################################################################
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
#linear algebra
import numpy as np
import pandas as pd
#helper modules
import acquire
#import prepare
#statistical tests
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
#visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plotly.express as px
import prepare

#########################################################- Explore Functions -#################################################################
def corr_plot(train)
    plt.figure(figsize = (12 , 8))
    train.corr()['wage_eur'].sort_values(ascending = False).plot(kind = 'barh', color = 'orange')
    plt.title('Relationship with Wages')
    plt.xlabel('Relationship')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

def corr_chart(train):
    """ allows us to see a correlation in percentage on the relationship with wage_eur"""
    features_cor =  train.corr()['wage_eur'].sort_values(ascending=False)
    features_cor = pd.DataFrame(features_cor)
    features_cor.tail(58)

def age_wage_plot(train)
    fig = px.box(train, x="age", y="wage_eur", points="all", animation_frame='year', color="league_name",
                    hover_name="league_name")
    fig.update_xaxes(categoryorder = 'mean ascending')
    fig.update_layout(title_text='Wage & Age', title_x=0.5)
    fig.show()

def seniority_wage_plot(train):
    fig = px.box(train, x="seniority", y="wage_eur", points="all", animation_frame='year', color="league_name",
                   hover_name="league_name")
    fig.update_xaxes(categoryorder = 'mean ascending')
    fig.update_layout(title_text='Wage & Seniority', title_x=0.5)
    fig.show()

def wage_reputation_plot(train):
    fig = px.box(train, x="international_reputation", y="wage_eur", points="all", animation_frame='year', color="league_name",
                    hover_name="league_name")
    fig.update_xaxes(categoryorder = 'mean ascending')
    fig.update_layout(title_text='Wages and Reputation', title_x=0.5)
    fig.show()
    
def wage_contract_chart(train):
    fig = px.box(train, x="club_contract_valid_until", y="wage_eur", points="all", animation_frame='year', color="club_contract_valid_until",
                   hover_name="league_name")
    fig.update_xaxes(categoryorder = 'mean ascending')
    fig.update_layout(title_text='Wages and Contract length', title_x=0.5)
    fig.show()

##################################################################################################################################################
