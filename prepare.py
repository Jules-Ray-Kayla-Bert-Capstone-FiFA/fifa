import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd

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
                        'league_level', 
                        'club_jersey_number', 
                        'club_loaned_from',
                        'nation_jersey_number',
                        'body_type',
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