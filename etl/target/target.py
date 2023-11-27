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
        """
        Retrieves the data from the specified Excel file.
        
        Returns:
        - df (pandas.DataFrame): The data read from the Excel file.
        """
        df = pd.read_excel(self.target_path, sheet_name='Pairs Analysis')
        return df
    
    def normalize(self, serie):
        """
        Normalizes the given series using sklearn's Normalizer.
        
        Args:
        - serie (list or numpy.ndarray): The series to be normalized.
        
        Returns:
        - serie_normalized (numpy.ndarray): The normalized series.
        - norm (sklearn.preprocessing.Normalizer): The Normalizer object used for normalization.
        """
        norm = Normalizer()
        serie_normalized = norm.fit_transform(np.array(serie).reshape(1, -1))
        serie_normalized = serie_normalized[0]
        return serie_normalized, norm
    
    def change(self, serie):
        """
        Calculates the percentage change between each element of the series and its previous element.
        
        Args:
        - serie (pandas.Series): The series to calculate the percentage change for.
        
        Returns:
        - percentage_change (pandas.Series): The percentage change series.
        """
        lag = serie.shift(1)
        percentage_change = (serie - lag) / lag
        return percentage_change
    
    def indexing(self, df, keep_price=True):
        """
        Performs indexing on the given DataFrame by calculating the percentage change for each series,
        summing the values for each date, and optionally keeping the original price values.
        
        Args:
        - df (pandas.DataFrame): The DataFrame to perform indexing on.
        - keep_price (bool, optional): Whether to keep the original price values. Defaults to True.
        
        Returns:
        - indexed_df (pandas.DataFrame): The indexed DataFrame.
        """
        df_copy = df.copy()
        for serie in self.series:
            values = self.change(df[df['SeriesID'] == serie]['Value'])
            df.loc[df['SeriesID'] == serie, 'Value'] = values
        df = df.groupby(['Date'])['Value'].sum().reset_index()

        if keep_price:
            for serie in self.series:
                df[f'Scrap_{serie}_Price'] = df_copy[df_copy['SeriesID'] == serie]['Value'].values
        df['Value'] = round(df['Value']/len(self.series), 4)
        return df
    
    def classify(self, serie):
        """
        Classifies the values in the given series into three categories based on quartiles.
        
        Args:
        - serie (numpy.ndarray): The series to be classified.
        
        Returns:
        - categories (pandas.Series): The categorized series.
        """
        quartiles = np.quantile(serie, self.quartiles)
        return pd.cut(serie, 
                      bins=[-np.inf, quartiles[0], quartiles[1], np.inf], 
                      labels=[0, 1, 2])
    

    def preprocess(self):
        """
        Performs data preprocessing on the target data.
        
        Returns:
        - df (pandas.DataFrame): The preprocessed DataFrame.
        """
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

        df = self.indexing(df, keep_price=True)
        df['Target'] = df['Value']
        df.drop(['Value'], axis=1, inplace=True)
        df.rename(columns={'Date':'date'}, inplace=True)
        df = df.resample('M', on='date').last().reset_index()
        df['date'] = df['date'].dt.date

        return df
    
    #############################
    # Data Loading Methods #
    
    def update(self):
        """
        Updates the target data by performing preprocessing and saving the cleaned data to a CSV file.
        """
        df = self.preprocess()
        df.to_csv(os.path.join(self.folder_path, 'target_clean.csv'), index=False)
