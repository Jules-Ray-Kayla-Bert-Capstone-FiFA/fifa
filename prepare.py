import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from sklearn.model_selection import train_test_split

def handle_missing_values(df):
    """this piece of code allows us to handle the missing data and get rid of it, both in the columns and in the rows(so that we can analize better)."""
    print ('Before dropping nulls, %d rows, %d cols' % df.shape)
    #drop collumns
    df = drop_columns(df)
    # add 0 values for goal keeper stats that are currently null
    df = goal_keeper_stats(df)
    #dropping.
    df = df.dropna()
    print('After dropping nulls. %d rows. %d cols' % df.shape)
    return df

def drop_columns(df):
    """ using this function to drop columns that i wont be using for exploration stage of this project"""
    # drop function to remove columns
    df = df.drop(columns = ['player_url', 
                        'long_name', 
                        'dob', 
                        'club_jersey_number', 
                        'club_loaned_from',
                        'nation_jersey_number',
                        'real_face',
                        'release_clause_eur',
                        'player_tags', 
                        'player_traits',
                        'ls', 'st', 'rs', 'lw', 'lf', 'cf',
                        'rf', 'rw', 'lam', 'cam', 'ram',
                        'lm', 'lcm', 'cm', 'rcm', 'rm',
                        'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
                        'lb','lcb', 'cb', 'rcb', 'rb',
                        'gk', 'player_face_url', 'club_logo_url',
                        'club_flag_url', 'nation_logo_url',
                        'nation_flag_url','mentality_composure',
                        'nation_position','nation_team_id',
                        ])
    return df

def goal_keeper_stats(df):
    #add 0 values to non goal keeper players
    df['goalkeeping_speed'].fillna("0", inplace = True)
    df['defending'].fillna("0", inplace = True)
    df['physic'].fillna("0", inplace = True)
    df['dribbling'].fillna("0", inplace = True)
    df['passing'].fillna("0", inplace = True)
    df['shooting'].fillna("0", inplace = True)
    df['shooting'].fillna("0", inplace = True)
    df['pace'].fillna("0", inplace = True)
    return df

def only_league_one(df):
    """ this function only lets us look at league 1 players"""
    # returns only league 1 players.
    df = df[df.league_level == 1.0]
    return df

def position_column(df):
    """ defines players positions """
    df['position'] = df.club_position.map({'RW':'Right Wing',
                                       'ST': 'Striker',
                                       'LW': 'Left Wing',
                                       'RCM': 'Right (off-centre) Centre Midfield',
                                       'GK': 'Goalkeeper',
                                       'CF': 'Centre Forward',
                                       'CDM': 'Centre Defensive Midfield',
                                       'LCB': 'Left Centre Back',
                                       'RDM':'Right Defensive Midfield',
                                       'RS': 'Right Striker',
                                       'LCM':'Left (off-centre) Centre Midfield',
                                       'SUB': 'Substitute',
                                       'CAM': 'Centre Attacking Midfield',
                                       'RCB':'Right Centre Back',
                                       'LDM':'Left Defensive Midfield',
                                       'LB': 'Left Back',
                                       'RB': 'Right Back',
                                       'LM': 'Left Midfield',
                                       'RM': 'Right Midfield',
                                       'CB': 'Centre Back',
                                       'LS': 'Left Striker',
                                       'RES': 'Reserves',
                                       'RWB': 'Right Wing Back',
                                       'LWB': 'Left Wing Back',
                                       'LAM': 'Left Attacking Midfield',
                                       'LF': 'Left Forward',
                                       'RAM': 'Right Attacking Midfield'
                        })
    return df

def column_positions(df):
        #changing the sequence of the columns
    sequence = ['sofifa_id', 'short_name', 'player_positions', 'overall', 'potential',
       'value_eur', 'wage_eur', 'age', 'height_cm', 'weight_kg',
       'club_team_id', 'club_name', 'league_name','nationality_id', 'nationality_name', 'league_level',
       'club_position', 'position', 'club_joined', 'club_contract_valid_until',
        'preferred_foot', 'weak_foot',
       'skill_moves', 'international_reputation', 'work_rate', 'pace',
       'shooting', 'passing', 'dribbling', 'defending', 'physic',
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'defending_marking_awareness', 'defending_standing_tackle',
       'defending_sliding_tackle', 'goalkeeping_diving',
       'goalkeeping_handling', 'goalkeeping_kicking',
       'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed',
       'year']
    #code to correct sequence
    df = df.reindex(columns=sequence)
    return df


def prepped_data(df):
    """ !!!!!!!!!!!!!!!     Please onlyuse this specific function for final notebook      !!!!!!!!!!!!!!!"""
    # handles missing values
    df = handle_missing_values(df)
    #returns only league 1 players
    df = only_league_one(df)
    #clarify position of players
    df = position_column(df)
    # change positions of columns
    df = column_positions(df)
    df = wrangle_fifa_data(df)
    print('After dropping leagues. %d rows. %d cols' % df.shape)
    return df

def split(df):
    train_and_validate, test = train_test_split(df, random_state=13, test_size=.15)
    train, validate = train_test_split(train_and_validate, random_state=13, test_size=.2)

    print('Train: %d rows, %d cols' % train.shape)
    print('Validate: %d rows, %d cols' % validate.shape)
    print('Test: %d rows, %d cols' % test.shape)
    
    return train, validate, test

def wrangle_fifa_data(df):
    #change numerical data to integers
    df.pace = df.pace.astype(int)
    df.shooting = df.shooting.astype(int)
    df.passing = df.passing.astype(int)
    df.dribbling = df.dribbling.astype(int)
    df.defending = df.defending.astype(int)
    df.physic = df.physic.astype(int)
    df.goalkeeping_speed = df.goalkeeping_speed.astype(int)
    #rename columns
    df = df.rename(columns = {'physic': 'physical',
                         'attacking_crossing':'crossing',
                         'attacking_finishing': 'finishing',
                         'attacking_heading_accuracy': 'heading_accuracy',
                         'attacking_short_passing': 'short_passing',
                         'attacking_volleys': 'volleys',
                         'skill_curve': 'curve',
                         'skill_fk_accuracy':'fk_accuracy',
                         'skill_long_passing': 'long_passing',
                         'skill_ball_control': 'ball_control',
                         'movement_acceleration': 'acceleration',
                         'movement_sprint_speed': 'sprint_speed',
                         'movement_agility': 'agility',
                         'movement_reactions': 'reactions',
                         'movement_balance': 'balance',
                         'power_shot_power': 'shot_power',
                         'power_jumping': 'jumping',
                         'power_stamina': 'stamina',
                         'power_strength' : 'strength',
                         'power_long_shots': 'long_shots',
                         'mentality_aggression': 'aggression',
                         'mentality_interceptions': 'interceptions',
                         'mentality_positioning': 'positioning',
                         'mentality_vision': 'vision',
                         'mentality_penalties': 'penalties',
                         'defending_marking_awareness': 'marking',
                         'defending_standing_tackle': 'standing_tackle',
                         'defending_sliding_tackle': 'sliding_tackle',
                         'goalkeeping_diving': 'gk_diving',
                         'goalkeeping_handling': 'gk_handling',
                         'goalkeeping_kicking': 'gk_kicking',
                         'goalkeeping_positioning': 'gk_positioning',
                         'goalkeeping_reflexes': 'gk_reflexes' ,
                         'goalkeeping_speed': 'gk_speed'
                                                })
    #add total wage column
    df['total_wage'] = df['value_eur'] + df['wage_eur']
    # change columns to datetime
    #df.club_joined = pd.to_datetime(df.club_joined)
    #create age bins players younger than 30 are considered younger, else, older
    df['age_bins'] = pd.cut(df['age'], bins = [0, 29, np.inf], labels = ['younger', 'older'])
    #create height bins
    df['height_bins'] = pd.cut(df['height_cm'], bins = 3, labels = ['short', 'medium', 'tall'])
    #create weight bins
    df['weight_bins'] = pd.cut(df['weight_kg'], bins = 3, labels = ['slim', 'average', 'heavy'])
    #only maintain league_level 1
    df = df[df.league_level == 1.0]
    #drop league_level
    df = df.drop(columns = ['league_level'])
    df = df.dropna()
    #expand the club_joined to get only the year from YYYY-MM-DD
    df['club_joined'] = df.club_joined.astype('str')
    df['year_joined'] = df.club_joined.str.split('-', expand = True)[0]
    #change joined year to int
    df.year_joined = df.year_joined.astype(int)
    #add seniority column
    df['seniority'] = df.year - df.year_joined
    #encoding
    df['league_encoded'] = df.league_name.map({'Argentina Primera División': 1,
                                              'English Premier League': 2,
                                              'USA Major League Soccer': 3,
                                              'French Ligue 1': 4,
                                              'Spain Primera Division': 5,
                                              'Italian Serie A': 6,
                                              'German 1. Bundesliga': 7,
                                              'Turkish Süper Lig': 8,
                                              'Portuguese Liga ZON SAGRES':9,
                                              'Mexican Liga MX': 10,
                                              'Holland Eredivisie': 11,
                                              'Colombian Liga Postobón': 12,
                                              'Belgian Jupiler Pro League': 13,
                                              'Polish T-Mobile Ekstraklasa': 14,
                                              'Saudi Abdul L. Jameel League': 15,
                                              'Swedish Allsvenskan': 16,
                                              'Japanese J. League Division': 17,
                                              'Norwegian Eliteserien': 18,
                                              'Chilian Campeonato Nacional': 19,
                                              'Danish Superliga': 20,
                                              'Korean K League': 21,
                                              'Scottish Premiership': 22,
                                              'Austrian Football Bundesliga': 23,
                                              'Rep. Ireland Airtricity League': 24,
                                              'Campeonato Brasileiro Série A': 25,
                                              'Swiss Super League': 26,
                                              'Russian Premier League': 27,
                                              'Australian Hyundai A-League': 28,
                                              'Chinese Super League': 29,
                                              'Romanian Liga I': 30,
                                              'Greek Super League': 31,
                                              'Ecuadorian Serie A': 32,
                                              'South African Premier Division': 33,
                                              'Paraguayan Primera División': 34,
                                              'Liga de Fútbol Profesional Boliian': 35,
                                              'Czech Republic Gambrinus Liga': 36,
                                              'Peruvian Primera División': 37,
                                              'Uruguayan Primera División': 38,
                                              'Venezuelan Primera División': 39,
                                              'Indian Super League': 40,
                                              'Ukrainian Premier League': 41,
                                              'Finnish Veikkausliiga': 42,
                                              'Croatian Prva HNL': 43,
                                              'UAE Arabian Gulf League': 44,
                                              'Hungarian Nemzeti Bajnokság I': 45,
                                              'Cypriot First Division':46,
                                              'Japanese J. League Division 1': 47,
                                              'Korean K League 1': 48,
                                              'Liga de Fútbol Profesional Boliviano': 49})
    return df
