# STANDARD LIBRARIES
import os
import warnings
warnings.filterwarnings("ignore")

# THIRD PARTY LIBRARIES
import numpy as npa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#category encoders
import category_encoders as ce

#statistical tests
from scipy import stats
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr

# default pandas decimal number display format
pd.options.display.float_format = '{:20,.2f}'.format

#sklearn imports
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

#import square root function
from math import sqrt

#import .py files
import acquire 
import prepare
import model

#import plotly express
import plotly.express as px