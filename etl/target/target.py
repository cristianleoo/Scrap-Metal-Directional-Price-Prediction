import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Normalizer

class Target:
    def __init__(self):
        self.folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'target')
        self.target_path = os.path.join(self.folder_path, 'NAT DATABASE.xlsx')
        self.series = [3, 9, 16]
        self.quartiles = [0.25, 0.75]
        self.norms = {}

    #############################
    # Data Preprocessing Methods #
    def get_data(self):    
        df = pd.read_excel(self.target_path, sheet_name='Pairs Analysis')
        return df
    
    def normalize(self, serie):
        norm = Normalizer()
        serie = norm.fit_transform(np.array(serie).reshape(1, -1))
        serie = serie[0]
        return serie, norm
    
    def change(self, serie):
        lag = serie.shift(1)
        return (serie - lag) / lag
    
    def indexing(self, df):
        for serie in self.series:
            values = self.change(df[df['SeriesID'] == serie]['Value'])
            df.loc[df['SeriesID'] == serie, 'Value'] = values
        df = df.groupby(['Date'])['Value'].sum().reset_index()
        df['Value'] = round(df['Value'], 4)
        return df
    
    def classify(self, serie):
        quartiles = np.quantile(serie, self.quartiles)
        return pd.cut(serie, 
                      bins=[-np.inf, quartiles[0], quartiles[1], np.inf], 
                      labels=[0, 1, 2])

    def preprocess(self):
        df = self.get_data()
        
        # Replace missing values with the previous value
        df.fillna(method='ffill', inplace=True, axis=0)

        # Use the melt function to move date columns into rows
        df = pd.melt(df, id_vars=['SeriesID', 'CatID', 'UOMID', 'Data Series', 'Category', 'Source', 'Unit'],
                    var_name='Date', value_name='Value')

        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Reorder columns
        df = df[['Date', 'SeriesID', 'CatID', 'UOMID', 'Data Series', 'Category', 'Source', 'Unit', 'Value']]

        # Subset the data to only include data from 2010 onwards
        df = df[df['Date'] >= '2010-01-01']

        # Replace missing values with 8s
        df['CatID'] = df['CatID'].fillna(8.0)

        # Keep only the series we are interested in (3, 9, 16)
        df = df[df['SeriesID'].isin(self.series)]

        # cols = ['Date', 'Data Series', 'Category', 'Source', 'Unit', 'Value']
        # df = df[cols]
        # df.reset_index(drop=True, inplace=True)
        df = self.indexing(df)
        df['Target'] = self.classify(df['Value'])
        return df
    
    #############################
    # Data Loading Methods #
    def update(self):
        df = self.preprocess()
        df.to_csv(os.path.join(self.folder_path, 'target_clean.csv'), index=False)
