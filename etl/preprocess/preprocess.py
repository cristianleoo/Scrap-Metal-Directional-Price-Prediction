import pandas as pd
import numpy as np
import os
import re

class Preprocess():
    def __init__(self, tick_list):
        """
        Initialize the Preprocess class.

        Parameters:
        - tick_list: list of tickers (stock symbols) to preprocess
        """
        self.data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/data'
        self.benzinga = None
        self.sentiment = None
        self.fred = None
        self.gmk = None
        self.weather = None
        self.tick_list = tick_list
        self.combination = None

        # Load Ferrous data
        try:
            self.ferrous = pd.read_csv(f'{self.data_path}/fred/ferrous.csv')
        except FileNotFoundError:
            self.ferrous = None
            print('Ferrous file not found')

        # Load Non Ferrous data
        try:
            self.non_ferrous = pd.read_csv(f'{self.data_path}/fred/non_ferrous.csv')
        except FileNotFoundError:
            self.non_ferrous = None
            print('Non Ferrous file not found')

        # Load Macro data
        try:
            self.macro = pd.read_csv(f'{self.data_path}/fred/macro.csv')
        except FileNotFoundError:
            self.macro = None
            print('Macro file not found')

        # Load Yahoo stock data
        try:
            self.stock = pd.read_csv(f'{self.data_path}/yahoo/yahoo.csv')
        except IndexError:
            self.stock = None
            print(f'Stock file for {self.tick} not found')

        # Load GMK data
        try:
            self.gmk = pd.read_csv(f'{self.data_path}/gmk/gmk.csv')
        except FileNotFoundError:
            self.gmk = None
            print(f'GMK file for {self.tick} not found')

        # Load weather data
        try:
            self.weather = pd.read_csv(f'{self.data_path}/weather/weather.csv')
        except FileNotFoundError:
            self.weather = None
            print(f'Weather file for {self.tick} not found')

        # Load target data
        try:
            self.target = pd.read_csv(f'{self.data_path}/target/target_clean.csv')
        except FileNotFoundError:
            self.target = None
            print(f'Target file for {self.tick} not found')

    def drop_missing_cols(self, df, threshold=0.2):
        """
        Drop columns from a DataFrame that have missing values above a given threshold.

        Parameters:
        - df: DataFrame to drop columns from
        - threshold: threshold for missing values (default: 0.2)

        Returns:
        - df: DataFrame with dropped columns
        """
        missing_cols = df.isna().sum() / len(df)
        missing_cols = missing_cols[missing_cols > threshold].index.tolist()
        df.drop(missing_cols, axis=1, inplace=True)
        return df

    def clean_benzinga(self):
        """
        Clean Benzinga data for each ticker in the tick_list.

        Prints a snapshot of the cleaned Benzinga data.
        """
        for tick in self.tick_list:
            try:
                benz = pd.read_csv(f"{self.data_path}/benzinga/{tick}.csv")
                benz = benz[['created', 'body']]
                benz['created'] = pd.to_datetime(benz['created'])
                benz = benz.resample('M', on='created')['body'].last().reset_index()
                benz = benz.rename(columns={'created': 'date', 'body': f'{tick}_benzinga'})

                if self.benzinga is None:
                    self.benzinga = benz
                else:
                    self.benzinga = pd.merge(self.benzinga, benz, on='date', how='outer')

            except FileNotFoundError:
                print(f'Benzinga file for {tick} not found')

        self.benzinga['date'] = self.benzinga['date'].dt.date
        print('Snapshot of Benzinga data:')
        print(self.benzinga.head())
        print(f"Size: {self.benzinga.shape}")

    def clean_sentiment(self, sentiment_model='gpt'):
        """
        Clean sentiment data for each ticker in the tick_list.

        Parameters:
        - sentiment_model: sentiment model to use (default: 'gpt')

        Prints a snapshot of the cleaned sentiment data.
        """
        for tick in self.tick_list:
            try:
                sentiment = pd.read_csv(f"{self.data_path}/{sentiment_model}/{tick}.csv")
                sentiment = sentiment[['created', 'sentiment']]
                sentiment['created'] = pd.to_datetime(sentiment['created'])
                sentiment = sentiment.resample('M', on='created')['sentiment'].mean().reset_index()
                sentiment['sentiment'] = sentiment['sentiment'].round(4)
                sentiment = sentiment.rename(columns={'created': 'date', 'sentiment': f'{tick}_sentiment'})

                if self.sentiment is None:
                    self.sentiment = sentiment
                else:
                    self.sentiment = pd.merge(self.sentiment, sentiment, on='date', how='outer')

            except FileNotFoundError:
                print(f'Finbert file for {tick} not found')

        self.sentiment['date'] = self.sentiment['date'].dt.date

        print('Snapshot of sentiment data:')
        print(self.sentiment.head())
        print(f"Size: {self.sentiment.shape}")

    def clean_stock(self):
        """
        Clean stock data.

        Prints a snapshot of the cleaned stock data.
        """
        self.stock['date'] = pd.to_datetime(self.stock['date']).dt.date

        zero_rows = set(np.where(self.stock.iloc[:10, 4:20] == 0)[0].tolist())
        inf_rows = set(np.where(np.isinf(self.stock.iloc[:, 4:20]))[0].tolist())
        neg_inf_rows = set(np.where(np.isinf(self.stock.iloc[:, 4:20]) & (self.stock.iloc[:, 4:20] < 0))[0].tolist())
        nan_rows = set(np.where(np.isnan(self.stock.iloc[:, 4:20]))[0].tolist())
        invalid_rows = zero_rows.union(inf_rows).union(neg_inf_rows).union(nan_rows)
        self.stock = self.stock.drop(list(invalid_rows)).reset_index(drop=True)

        print('Snapshot of stock data:')
        print(self.stock.head())
        print(f"Size: {self.stock.shape}")

    def clean_fred(self):
        """
        Clean FRED (Federal Reserve Economic Data) data.

        Prints a snapshot of the cleaned FRED data.
        """
        fred = pd.merge(self.ferrous, self.non_ferrous, on='date', how='left', suffixes=('_ferrous', '_non_ferrous'))
        fred['date'] = pd.to_datetime(fred['date'])
        fred = fred.resample('M', on='date').last().reset_index()
        self.macro['date'] = pd.to_datetime(self.macro['date'])
        fred = pd.merge(fred, self.macro, on='date', how='left')
        fred['date'] = fred['date'].dt.date
        self.fred = fred
        print('Snapshot of macro data:')
        print(fred.head())
        print(f"Size: {fred.shape}")

    def clean_gmk(self):
        """
        Clean GMK (Global Market News) data.

        Prints a snapshot of the cleaned GMK data.
        """
        self.gmk['Date'] = pd.to_datetime(self.gmk['Date']).dt.date
        self.gmk = self.gmk[['Date', 'Title', 'Content']]
        self.gmk = self.gmk.rename(columns={'Date': 'date', 'Title': 'title', 'Content': 'content'})
        print('Snapshot of GMK data:')
        print(self.gmk.head())
        print(f"Size: {self.gmk.shape}")

    def clean_weather(self):
        """
        Clean weather data.

        Prints a snapshot of the cleaned weather data.
        """
        weather = self.weather
        weather.rename(columns={'type': 'Month', 'option': 'area'}, inplace=True)
        weather = weather[weather['Month'] != 'Annual']
        weather['date'] = pd.to_datetime(weather['Year'].astype(str) + '-' + weather['Month'].astype(str))
        weather['date'] = weather['date'] + pd.offsets.MonthEnd()
        weather['temperature'] = weather['temperature'].apply(lambda x: float(x) if x!='M' else np.nan)
        weather = weather.groupby(['area', 'date'])['temperature'].mean().reset_index()
        weather = weather.pivot(index='date', columns='area', values='temperature')
        weather.dropna(axis=1, thresh=0.95, inplace=True)
        weather.reset_index(inplace=True)
        weather['date'] = weather['date'].dt.date
        weather.rename(columns=lambda x: re.sub(r'\W+', '_', x.strip()), inplace=True)
        
        self.weather = weather
        print('Snapshot of weather data:')
        print(weather.head())
        print(f"Size: {weather.shape}")

    def merge_table(self):
        """
        Merge all cleaned data tables.

        Prints a snapshot of the merged data table.
        """
        self.combination = pd.merge(self.stock, self.fred, on='date', how='left')

        if self.benzinga is not None:
            self.combination = pd.merge(self.combination, self.benzinga, on='date', how='left')

        if self.sentiment is not None:
            self.combination = pd.merge(self.combination, self.sentiment, on='date', how='left')

        cols_failure = ['GDXJ_x', 'GDXJ_y', 'IYT_y']
        for col in cols_failure:
            try:
                self.combination.drop(col, axis=1, inplace=True)
            except KeyError:
                pass
        
        self.combination = pd.merge(self.combination, self.weather, on='date', how='left')

        self.target['date'] = pd.to_datetime(self.target['date']).dt.date
        self.combination = pd.merge(self.combination, self.target, on='date', how='inner')
        self.combination = self.drop_missing_cols(self.combination, threshold=0.2)
        print(self.combination.head())

    def export_to_csv(self):
        """
        Export the merged data table to a CSV file.
        """
        if not os.path.exists(f"{self.data_path}/preprocessed"):
            os.makedirs(f"{self.data_path}/preprocessed")
            print(f"Created directory {self.data_path}/preprocessed")
        self.combination.to_csv(f"{self.data_path}/preprocessed/data.csv", index=False)
    
    def main(self, benzinga=False, sentiment_model='gpt'):
        """
        Main function to preprocess the data.

        Parameters:
        - benzinga: whether to clean Benzinga data (default: False)
        - sentiment_model: sentiment model to use (default: 'gpt')
        """
        if benzinga:
            self.clean_benzinga()
        else:
            self.clean_sentiment(sentiment_model=sentiment_model)
        self.clean_stock()
        self.clean_fred()
        self.clean_weather()
        self.merge_table()
        self.export_to_csv()

if __name__ == '__main__':
    tick_list = ['NUE', 'STLD', 'NSC', 'CLF', 'AA']
    preprocess = Preprocess(tick_list)
    # preprocess.main(benzinga=False, sentiment_model='gpt')
    preprocess.clean_weather()