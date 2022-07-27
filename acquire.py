import pandas as pd

def get_fifa_data():
    """ The original excel file can be found by going to the following website (https://www.kaggle.com/bryanb/fifa-player-stats-database)"""
    #looking for csv file
    filename = "fifa.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the excel file dataframe, verify that they have a year column assigned.
        # import excel sheets frame from local file
        df = pd.read_excel('fifa_male.xlsx')
        # Read excel file
        xlsx = pd.ExcelFile('fifa_male.xlsx')
        # import data for year 2022
        df1 = pd.read_excel(xlsx, 'FIFA 22')
        # import data for year 2021
        df2 = pd.read_excel(xlsx, 'FIFA 21')
        # import data for year 2020
        df3 = pd.read_excel(xlsx, 'FIFA 20')
        # import data for year 2019
        df4 = pd.read_excel(xlsx, 'FIFA 19')
        # import data for year 2018
        df5 = pd.read_excel(xlsx, 'FIFA 18')
        # import data for year 2017
        df6 = pd.read_excel(xlsx, 'FIFA 17')
        # import data for year 2016
        df7 = pd.read_excel(xlsx, 'FIFA 16')
        # import data for year 2015
        df8 = pd.read_excel(xlsx, 'FIFA 15')
        "It is important that we add a year column so that our data is not contaminated across the years and to make exploration easier"
        # now to add a year column to each csv file
        df1['year'] = 2022
        df2['year'] = 2021
        df3['year'] = 2020
        df4['year'] = 2019
        df5['year'] = 2018
        df6['year'] = 2017
        df7['year'] = 2016
        df8['year'] = 2015
        # combining dataframes together
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
        # in this case we will not be making a csv file but if you wish to so use the following code to make into a csv file
        """df.to_csv(filename, index = False)"""
        # Return the dataframe to the calling code
        return df