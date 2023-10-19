import pandas as pd
import numpy as np
import os
import datetime

class Preprocess():
    def __init__(self, tick_list):
        self.benzinga = None
        self.finbert_sentiment = None
        self.macro = None
        self.fred = None
        self.tick_list = tick_list
        self.stock = None
        self.combination = None
        # try:
        #     self.benzinga = pd.read_csv(os.path.join(os.getcwd(), 'data', self.tick, 'benzinga_ratings.csv'))
        # except FileNotFoundError:
        #     print('Benzinga file not found')

        try:
            self.ferrous = pd.read_csv(os.path.join(os.getcwd(), 'data/fred/ferrous.csv'))
        except FileNotFoundError:
            self.ferrous = None
            print('Macro file not found')

        try:
            self.non_ferrous = pd.read_csv(os.path.join(os.getcwd(), 'data/fred/non_ferrous.csv'))
        except FileNotFoundError:
            self.non_ferrous = None
            print('Macro file not found')
        # try:
        #     self.youtube = pd.read_csv(os.path.join(os.getcwd(), 'data','youtube_with_ratings.csv'))
        # except FileNotFoundError:
        #     print('YouTube file not found')

        # get Yahoo data
        try:
            self.stock = pd.read_csv(os.path.join(os.getcwd(), 'data/yahoo/yahoo.csv'))
        except IndexError:
            self.stock = None
            print(f'Stock file for {self.tick} not found')

        try:
            self.target = pd.read_csv(os.path.join(os.getcwd(), 'data/target/target_clean.csv'))
        except FileNotFoundError:
            self.target = None
            print(f'Target file for {self.tick} not found')
        # try:
        #     earning_file = glob.glob(os.path.join(os.getcwd(), 'data', self.tick, 'earnings.csv'))[0]
        #     self.earning = pd.read_csv(earning_file)
        # except IndexError:
        #     print(f'Earning file for {self.tick} not found')

    def clean_benzinga(self):
        self.benzinga = self.benzinga[['created', 'benz_rate']]
        self.benzinga.dropna(subset=['benz_rate'], inplace=True)
        self.benzinga['benz_rate'] = self.benzinga['benz_rate'].astype(int)
        self.benzinga = self.benzinga.groupby('created')['benz_rate'].mean().reset_index()
        self.benzinga['benz_rate'] = self.benzinga['benz_rate'].round(4)
        self.benzinga = self.benzinga.rename(columns={'created': 'date'})
        self.benzinga['date'] = pd.to_datetime(self.benzinga['date']).dt.date
        print('Snapshot of benzinga data:')
        print(self.benzinga.head())
        print(f"Size:{self.benzinga.shape}")

    def clean_finbert(self):
        for tick in self.tick_list:
            try:
                finbert = pd.read_csv(f"{os.getcwd()}/data/finbert/{tick}.csv")
                finbert = finbert[['created', 'sentiment']]
                # finbert = finbert.groupby('created')['sentiment'].mean().reset_index()
                finbert['created'] = pd.to_datetime(finbert['created'])
                finbert = finbert.resample('M', on='created')['sentiment'].mean().reset_index()
                finbert['sentiment'] = finbert['sentiment'].round(4)
                finbert = finbert.rename(columns={'created': 'date', 'sentiment': f'{tick}_sentiment'})
            except FileNotFoundError:
                print(f'Finbert file for {tick} not found')
            if self.finbert_sentiment is None:
                self.finbert_sentiment = finbert
            else:
                self.finbert_sentiment = pd.merge(self.finbert_sentiment, finbert, on='date', how='outer')

        self.finbert_sentiment.fillna(method='ffill', inplace=True)
        #self.finbert_sentiment['date'] = pd.to_datetime(self.finbert_sentiment['date'])
        
        self.finbert_sentiment['date'] = self.finbert_sentiment['date'].dt.date
        self.finbert_sentiment.fillna(method='bfill', inplace=True)
        print('Snapshot of finbert data:')
        print(self.finbert_sentiment.head())
        print(f"Size:{self.finbert_sentiment.shape}")
    
    def clean_stock(self):
        self.stock['date'] = pd.to_datetime(self.stock['date']).dt.date
        self.stock.fillna(method='ffill', inplace=True)

        zero_rows = set(np.where(self.stock.iloc[:10, 4:20] == 0)[0].tolist())
        inf_rows = set(np.where(np.isinf(self.stock.iloc[:, 4:20]))[0].tolist())
        neg_inf_rows = set(np.where(np.isinf(self.stock.iloc[:, 4:20]) & (self.stock.iloc[:, 4:20] < 0))[0].tolist())
        nan_rows = set(np.where(np.isnan(self.stock.iloc[:, 4:20]))[0].tolist())
        invalid_rows = zero_rows.union(inf_rows).union(neg_inf_rows).union(nan_rows)
        self.stock = self.stock.drop(list(invalid_rows)).reset_index(drop=True)

        print('Snapshot of stock data:')
        print(self.stock.head())
        print(f"Size:{self.stock.shape}")

    def clean_fred(self):
        fred = pd.merge(self.ferrous, self.non_ferrous, on='date', how='left', suffixes=('_ferrous', '_non_ferrous'))
        fred['date'] = pd.to_datetime(fred['date'])
        fred = fred.resample('M', on='date').last().reset_index()
        fred['date'] = fred['date'].dt.date
        self.fred = fred
        print('Snapshot of macro data:')
        print(fred.head())
        print(f"Size:{fred.shape}")

    def merge_table(self):
        self.combination = pd.merge(self.stock, self.fred, on='date', how='left')

        if self.benzinga is not None:
            self.combination = pd.merge(self.combination, self.benzinga, on='date', how='left')
            # Fill NA values using backward fill method for dates when market is closed
            columns_to_fill = [col for col in self.stock.columns if col not in ['date', 'benz_rate']]
            self.combination[columns_to_fill] = self.combination[columns_to_fill].fillna(method='bfill')
            # # Fill NA values using forward fill method for dates when no news created when market is open
            self.combination['benz_rate'] = self.combination['benz_rate'].fillna(method='ffill')
            self.combination['benz_rate'] = self.combination.groupby(['close', 'volume', 'day'])['benz_rate'].transform('mean')
            self.combination['benz_rate'] = self.combination['benz_rate'].round(3)

        if self.finbert_sentiment is not None:
            self.combination = pd.merge(self.combination, self.finbert_sentiment, on='date', how='left')
        self.combination.fillna(method='ffill', inplace=True)
        self.combination.fillna(method='bfill', inplace=True)
        self.combination['target'] = self.target['Target']

    def export_to_csv(self):
        if not os.path.exists(f"{os.getcwd()}/data/preprocessed"):
            os.makedirs(f"{os.getcwd()}/data/preprocessed")
            print(f"Created directory {os.getcwd()}/data/preprocessed")
        self.combination.to_csv(f"{os.getcwd()}/data/preprocessed/data.csv", index=False)
    
    def main(self):
        if self.benzinga is not None:
            self.clean_benzinga()
        self.clean_finbert()
        self.clean_stock()
        self.clean_fred()
        self.merge_table()
        self.export_to_csv()