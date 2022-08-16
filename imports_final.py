#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#linear algebra
import numpy as np
import pandas as pd
#import geopandas as gpd

#helper modules
import acquire
import prepare_final 
import model
import final_modeling
import tests
import visuals_final
#import explore


#statistical tests
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau


from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#visualizations
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib import rcParams
import plotly.express as px
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
#imports to show interactive visuals on github
import plotly.io as pio
pio.renderers

# renderer="svg"
# fig.show(renderer="svg")

#display max columns
pd.set_option('display.max_columns', None)

#use this format specifier to show 20 total numbers, with 2 behind decimal
#pd.options.display.float_format = '{:20.2f}'.format 

sns.set_style("white")