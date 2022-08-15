#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#linear algebra
import numpy as np
import pandas as pd

#helper modules
import acquire
import prepare
import model
import final_modeling
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
from matplotlib import rcParams
import plotly.express as px


#display max columns
pd.set_option('display.max_columns', None)

#use this format specifier to show 20 total numbers, with 2 behind decimal
#pd.options.display.float_format = '{:20.2f}'.format 

sns.set_style("white")

#acquire
df = acquire.get_fifa_data()

#prepare
df = prepare.prepped_data(df)

#train, test, split
train, validate, test = prepare.split(df)

########################################################################################################################

def age_stats():
    #set alpha
    α = 0.05

    #perform test
    coef, p = kendalltau(train.age, train.wage_eur)

    #evaluate coefficient and p-value
    print(f'τ: {coef:.3f}\nP-Value: {p:.3f}')

    #evaluate if p < α
    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def seniority_stats():
    #set alpha
    α = 0.05

    #perform test
    coef, p = kendalltau(train.seniority, train.wage_eur)

    #evaluate coefficient and p-value
    print(f'τ: {coef:.3f}\nP-Value: {p:.3f}')

    #evaluate if p < α
    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def league_stats():
    α = 0.05
    #setup crosstab
    observed = pd.crosstab(train.league_name, train.wage_eur)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    round(p ,3)

    #print p-value
    print(f'P-Value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null.')

def league_stats_ia():
    #set alpha
    α = 0.05

    #get sample
    isa_sample = train[train.league_name == 'Italian Serie A'].wage_eur

    #get mean
    overall_mean = train.wage_eur.mean()

    #perform test
    t, p = stats.ttest_1samp(isa_sample, overall_mean)

    #print p-value
    print(f'P-Value: {p/2:.3f}')

    #evaluate if mean of ISA wages is significantly higher than all wages, is p/2 < a and t > 0?
    if p/2 < α and t > 0:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')


def club_stats():    

    #set alpha
    α = 0.05

    #get sample
    bar_sample = train[train.club_name == 'FC Barcelona'].wage_eur

    #get mean
    overall_mean = train.wage_eur.mean()

    #perform test
    t, p = stats.ttest_1samp(bar_sample, overall_mean)

    #print p-value
    print(f'P-Value: {p/2:.3f}')

    #evaluate if mean of Barcelona wages is significantly higher than all wages, is p/2 < a and t > 0?
    if p/2 < α and t > 0:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def body_stats():
    α = 0.05
    #setup crosstab
    observed = pd.crosstab(train.body_type, train.wage_eur)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    round(p ,3)

    #print p-value
    print(f'P-Value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null.')

def body_tt():

    #set alpha
    α = 0.05

    #get sample
    nor_sample = train[train.body_type == 'Normal (170-185)'].wage_eur

    #get mean
    overall_mean = train.wage_eur.mean()

    #perform test
    t, p = stats.ttest_1samp(nor_sample, overall_mean)

    #print p-value
    print(f'P-Value: {p/2:.3f}')

    #evaluate if mean wages of those with normal body types is significantly higher than other types, is p/2 < a and t > 0?
    if p/2 < α and t > 0:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def body_tt_unique():

    #Levene's test confirmed unequal variance, set to False

    #set alpha
    α = 0.05

    #perform test
    t, p = stats.ttest_ind(train[train.body_type == 'Normal (170-185)'].wage_eur,  train[train.body_type == 'Unique'].wage_eur, equal_var = False)

    #print p-value
    print(f'P-Value: {p/2:.3f}')

    #evaluate if mean wages of those with unique body types is significantly less than mean wage of players with normal body types, is p/2 < a and t < 0?

    if p/2 < α and t < 0:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def body_height_k():
    #height is normally distributed, use Pearson's
    #set alpha
    α = 0.05

    #perform test
    coef, p = kendalltau(train.height_cm, train.wage_eur)

    #evaluate coefficient and p-value
    print(f'τ: {coef:.3f}\nP-Value: {p:.3f}')

    #evaluate if p < α
    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def body_weight_k():
    #set alpha
    α = 0.05

    #perform test
    coef, p = kendalltau(train.weight_kg, train.wage_eur)

    #evaluate coefficient and p-value
    print(f'τ: {coef:.3f}\nP-Value: {p:.3f}')

    #evaluate if p < α
    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null hypothesis.')

def nationality():
    α = 0.05

    #setup crosstab
    observed = pd.crosstab(train.nationality_name, train.wage_eur)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    round(p ,3)

    #print p-value
    print(f'P-Value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null.')

def reputation():

    α = 0.05
    #setup crosstab
    observed = pd.crosstab(train.international_reputation, train.wage_eur)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    round(p ,3)

    #print p-value
    print(f'P-Value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null.')


def seniority():
    α = 0.05
    #setup crosstab
    observed = pd.crosstab(train.seniority, train.wage_eur)

    chi2, p, degf, expected = stats.chi2_contingency(observed)
    round(p ,3)

    #print p-value
    print(f'P Value: {p:.3f}')

    if p < α:
        print('Reject the null hypothesis.')
    else:
        print('Fail to reject the null.')

