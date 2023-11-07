from __future__ import annotations
from etl.etl import ETL
import pandas as pd
import yfinance as yf
from stockstats import StockDataFrame as Sdf
import os

class Yahoo(ETL):
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API
    """

    def __init__(self, tick_list, etfs=None, period='monthly', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tick_list = tick_list
        self.df = None
        self.etfs = etfs
        self.period = period
    
    def add_etfs(self, df):
        etfs = self.etfs
        etf_data = None

        for etf in etfs:
            if self.period == 'monthly':
                temp_data = yf.download(etf, start=self.start_day, end=self.end_day, interval='1mo')["Close"]
            else:
                temp_data = yf.download(etf, start=self.start_day, end=self.end_day)["Close"]
            temp_data = temp_data.reset_index()[['Date', 'Close']]
            temp_data = temp_data.rename(columns={"Close": etf})

            if etf_data is None:
                etf_data = temp_data
            else:
                etf_data = pd.merge(etf_data, temp_data, on="Date", how="left")#.drop_duplicates()
            print(f"{etf} added to dataset")

        print(etf_data.head())
        #df["date"] = pd.to_datetime(df["date"])
        etf_data.reset_index(inplace=True, drop=True)
        etf_data["date"] = etf_data["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        merged_df = pd.merge(df, etf_data, on="date", how="left")
        # merged_df = merged_df.fillna(method='ffill')
        self.df = merged_df
        return merged_df
    
    def fecth_ticker(self, ticker):
        if self.period == 'monthly':
            df = yf.download(ticker, start=self.start_day, end=self.end_day, interval='1mo')
        else:
            df = yf.download(ticker, start=self.start_day, end=self.end_day)
        df.reset_index(inplace=True)
        df["Date"] = df["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        # df.fillna(0, inplace=True)

        try:
            df.columns = df.columns.str.lower()
            # use adjusted close price instead of close price
            df["close"] = df["adj close"]
            # drop the adjusted close price column
            df = df.drop(labels="adj close", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        df = df[['date', 'close']]
        df.rename(columns={'close': ticker}, inplace=True)

        return df
    
    def drop_missing_cols(self, df, threshold=0.2):
        missing_cols = df.isna().sum() / len(df)
        missing_cols = missing_cols[missing_cols > threshold].index.tolist()
        df.drop(missing_cols, axis=1, inplace=True)
        return df
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        # print(self.ticker)
        # df = yf.download(
        #     self.ticker, start=self.start_day, end=self.end_day, proxy=proxy
        # )
        # df.reset_index(inplace=True)
        # df["Date"] = df["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
        # df.fillna(0, inplace=True)
        for ticker in self.tick_list:
            df = self.fecth_ticker(ticker)
            if self.df is None:
                self.df = df
            else:
                self.df = pd.merge(self.df, df, on="date", how="left")
        if self.etfs:
            df = self.add_etfs(self.df)   

        # df = df.fillna(method='ffill')

        # df.dropna(how='all', inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        # convert date to standard string format, easy to filter
        df = df.resample("M", on="date").last().reset_index()
        df["date"] = df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        #data_df = data_df.dropna()
        df = df.reset_index(drop=True)
        print("Shape of DataFrame: ", df.shape)
        
        df = df.sort_values(by="Date").reset_index(drop=True)
        df.drop(labels="Date", axis=1, inplace=True)

        df = self.drop_missing_cols(df, threshold=0.2)

        self.df = df
        # keep the value of the end of each month
        return df

    # def add_bond(self):
    #     etfs = ["TLT", "IEF", "SHY"]
    #     etf_data = None

    #     for etf in etfs:
    #         temp_data = yf.download(etf, start=self.start_day, end=self.end_day)["Close"]
    #         temp_data = temp_data.reset_index().rename(columns={"Date": "date", "Close": etf})

    #         if etf_data is None:
    #             etf_data = temp_data
    #         else:
    #             etf_data = pd.merge(etf_data, temp_data, on="date", how="left")

    #     self.df["date"] = pd.to_datetime(self.df["date"])
    #     merged_df = pd.merge(self.df, etf_data, on="date", how="left")
    #     merged_df = merged_df.fillna(method='ffill')
    #     self.df = merged_df
    #     return merged_df


    def export_as_csv(self):
        exclude_columns = ['date']
        self.df.loc[:, ~self.df.columns.isin(exclude_columns)] = self.df.loc[:, ~self.df.columns.isin(exclude_columns)].round(3)


        if not os.path.exists(f'{os.getcwd()}/data/yahoo'):
            os.makedirs(f'{os.getcwd()}/data/yahoo')
            print(f"Created directory {os.getcwd()}/data/yahoo")
        self.df.to_csv(f'{os.getcwd()}/data/yahoo/yahoo.csv', index=False)
        print(f"data/yahoo.csv created!")