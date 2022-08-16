from imports_final import *

#visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import plotly.express as px
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
#imports to show interactive visuals on github
import plotly.io as pio
pio.renderers

#acquire
df = acquire.get_fifa_data()

#prepare
df = prepare_final.prepped_data(df)

#train, test, split
train, validate, test = prepare_final.split(df)

def viz_1():
    pop_clubs = ('Manchester City', 'FC Barcelona',  'Chelsea', 'Real Madrid CF', 'Juventus', 'Manchester United')


    fig = px.scatter(train.loc[train['club_name'].isin(pop_clubs)],
                    x='overall', 
                    y='wage_eur', 
                    color='club_name', 
                    title='Salary and Overall',
                    hover_name = 'club_name',
                    labels = {'wage_eur' : 'Salary (€)',
                            'club_name_yr_sum': 'Club Sum',
                            'club_name': 'Club',
                            'overall': 'Overall'},
                    width = 1000,
                    height = 700)
    fig.show(renderer = 'svg')

def viz_2():
    pop_clubs = ('Manchester City', 'FC Barcelona',  'Chelsea', 'Real Madrid CF', 'Juventus', 'Manchester United')


    fig = px.scatter(train.loc[train['club_name'].isin(pop_clubs)],
                    x='potential', 
                    y='wage_eur', 
                    color='club_name', 
                    title='Salary and Potential',
                    hover_name = 'club_name',
                    labels = {'wage_eur' : 'Salary (€)',
                            'club_name_yr_sum': 'Club Sum',
                            'club_name': 'Club',
                            'potential': 'Potential'},
                    width = 1000,
                    height = 700)
    fig.show(renderer = 'svg')

def viz_3():
    pop_clubs = ('Manchester City', 'FC Barcelona',  'Chelsea', 'Real Madrid CF', 'Juventus', 'Manchester United')


    fig = px.scatter(train.loc[train['club_name'].isin(pop_clubs)],
                    x='age', 
                    y='wage_eur', 
                    color='club_name', 
                    title='Salary and Age',
                    hover_name = 'club_name',
                    labels = {'wage_eur' : 'Salary (€)',
                            'club_name_yr_sum': 'Club Sum',
                            'club_name': 'Club',
                            'age': 'Age'},
                    width = 1000,
                    height = 700)
    fig.show(renderer = 'svg')

def viz_4():
    fig = px.box(train, x = "position", 
             y = "wage_eur", 
             points = "all", 
             animation_frame = 'year', 
             color = "league_name",
             hover_name = "league_name",
             labels = {"wage_eur": "Salary (€)",
                     "position": "Position",
                     "league_name": "League"
                 },
            width = 1200,
            height = 800)

    fig.update_xaxes(categoryorder = 'mean ascending')

    fig.update_layout(title_text = 'Wages and Positions')

    fig.show(renderer = 'svg')

def viz_5():
    fig = px.scatter(train.sample(n=20000, replace=False, random_state=123).sort_index(), 
                 x='league_name', 
                 y='wage_eur', 
                 color='league_yr_sum', 
                 title='Does the League a Player is with impact Salary?',
                hover_name = 'league_name',
                labels = {'wage_eur' : 'Salary (€)',
                         'league_yr_sum': 'League Sum',
                         'league_name': 'League'},
                width = 1000,
                height = 700)

    fig.update_xaxes(categoryorder='total descending')
    fig.show(renderer = 'svg')

def viz_6():
    clubs = train.groupby('club_name')['wage_eur'].sum().reset_index()
    clubs = clubs.sort_values('wage_eur', ascending=False).head(20)

    fig = px.histogram(clubs, 
                    x='club_name', 
                    y='wage_eur',
                    hover_name = 'club_name',
                    labels = {'sum of wage_eur' : 'Salary Totals by Club',
                            'club_name': 'Club'},
                    width = 1000,
                    height = 700)

    fig.show(renderer="svg")

def viz_7():
    clubs = train.groupby(['club_name', 'body_type'])['wage_eur'].sum().reset_index()
    clubs = clubs.sort_values('wage_eur', ascending=False).head(20)

    fig = px.histogram(clubs, 
                    x='body_type', 
                    y='wage_eur',
                    color = 'club_name',
                    labels = {'sum of wage_eur' : 'Salary Totals by Club',
                            'club_name': 'Club',
                            'body_type': 'Body Type'},
                    width = 1000,
                    height = 700)
    fig.show(renderer = 'svg')

def viz_8():
    pop_clubs = ('Manchester City', 'FC Barcelona',  'Chelsea', 'Real Madrid CF', 'Juventus', 'Manchester United')


    fig = px.scatter(train.loc[train['club_name'].isin(pop_clubs)],
                    x='height_cm', 
                    y='wage_eur', 
                    color='club_name', 
                    title='Height and Salary',
                    hover_name = 'short_name',
                    labels = {'wage_eur' : 'Salary (€)',
                            'club_name_yr_sum': 'Club Sum',
                            'club_name': 'Club',
                            'height_cm': 'Height(cm)'},
                    width = 1000,
                    height = 700)
    fig.show(renderer = 'svg')

def viz_9():
    pop_clubs = ('Manchester City', 'FC Barcelona',  'Chelsea', 'Real Madrid CF', 'Juventus', 'Manchester United')


    fig = px.scatter(train.loc[train['club_name'].isin(pop_clubs)],
                    x='weight_kg', 
                    y='wage_eur', 
                    color='field_position', 
                    title='Weight and Salary',
                    hover_name = 'short_name',
                    labels = {'wage_eur' : 'Salary (€)',
                            'club_name': 'Club',
                            'weight_kg': 'Weight(kg)',
                            'field_position': 'Positions'},
                    width = 1000,
                    height = 700)
    fig.show(renderer = 'svg')

def viz_10():
    nations = ('Spain', 'France',  'Italy', 'Belgium', 'Columbia', 'Sweden', 'United States', 'Austria',\
           'Swizerland', 'Senegal', "Côte d'Ivoire", 'Republic of Ireland', 'Australia', 'Ecudor'\
          'Bosnia and Herzegovina', 'Ukraine', 'Tunisia', 'Congo DR')

    fig = px.histogram(train.loc[train['nationality_name'].isin(nations)].sort_index(), 
                    x='nationality_name', 
                    y='wage_eur', 
                    template='seaborn', 
                    color='league_name', 
                    title='Does nationality impact salary?',
                    labels = {'league_name': 'League',
                                'nationality_name': 'Country'},
                    width = 1000,
                    height = 700)

    fig.update_xaxes(categoryorder = 'total descending')
    fig.show(renderer ="svg")

def viz_11():
    # graphs age vs salary on boxplot
    fig = px.box(train.sample(n=20000, replace=False, random_state=123).sort_index(), 
                x = 'international_reputation', 
                y = 'wage_eur', 
                color = 'international_reputation',
                hover_name = 'wage_eur', 
                template='presentation',
                title='International and Salary',
                labels = {'wage_eur': 'Salary (€)',
                        'international_reputation':'International Reputation'},
                width = 1000,
                height = 700)

    fig.update_layout(showlegend=False)
    fig.show(renderer ="svg")

def viz_12():
    fig = px.box(train, x="club_contract_valid_until", 
             y = "wage_eur", 
             points = "all", 
             animation_frame = 'year', 
             color = "league_name",
             hover_name = "league_name",
             labels = {"wage_eur": "Salary (€)",
                     "club_contract_valid_until": "Contract Year",
                     "league_name": "League"
                 },
            width = 1200,
            height = 800)


    fig.update_xaxes(categoryorder = 'mean ascending')
    fig.update_layout(title_text = 'Wage Per Position and Contract')
    fig.show(renderer = 'svg')

