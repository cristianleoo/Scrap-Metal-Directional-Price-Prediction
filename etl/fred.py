from fredapi import Fred
from etl.etl import ETL
import os
import requests
import pandas as pd

class Fredapi(ETL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices = ["PAYEMS", "FEDFUNDS", "UNRATE", "PPIACO", "CPIAUCSL",
                        "PCE", "INDPRO", "UMCSENT", "RSAFS", "HOUST", "CSUSHPINSA"],
        self.metals_url = 'https://api.stlouisfed.org/fred/series/observations'
        self.series_ids = {
            'ferrous_price': 'WPU10121501',
            'non_ferrous_price': 'WPS102',
            'ferrous_inventory': 'A31CTI',
            'alluminum_nonferrous_inventory': 'AANMTI',
            'ferrous_orders': 'A31CNO',
            'non_ferrous_orders': 'AANMNO'
        }
        self.frequency = 'm' # monthly

    #############################################
    # Fetch macro data from FRED

    def fetch_macro_data(self):
        fred = Fred(api_key=self.api_keys["Fred"])
        dfs = []

        # Mapping of series IDs to their actual names
        names = {
            "PAYEMS": "NFP", # Non-Farm Payrolls
            "CPIAUCSL": "CPI", # Consumer Price Index
            "FEDFUNDS": "InterestRate", # Effective Federal Funds Rate
            "UNRATE": "UnemploymentRate",
            "PPIACO": "PPI", # Producer Price Index
            "PCE": "PCE", # Personal Consumption Expenditures
            "INDPRO": "IPI", # Industrial Production Index
            "UMCSENT": "ConsumerSentiment",
            "RSAFS": "RetailSales",
            "HOUST": "HousingStarts",
            "CSUSHPINSA": "HPI", # S&P/Case-Shiller U.S. National Home Price Index
        }

        for index in self.indices[0]: # Note: self.indices is a tuple with a single list
            series = fred.get_series(index, self.start_day, self.end_day)
            series = series.resample("D").ffill().reset_index()
            series.columns = ["date", index]
            dfs.append(series)

        # Merge all dataframes on 'date'
        macro_df = dfs[0]
        for df in dfs[1:]:
            macro_df = macro_df.merge(df, on='date', how='outer')

        # Rename the columns
        macro_df.rename(columns=names, inplace=True)

        macro_df = macro_df.round(4)
        macro_df.to_csv(f'{os.getcwd()}/data/macro.csv', index=False)
        print('macro.csv created')
        return macro_df

    #############################################
    # Fetch ferrous and non-ferrous metal prices

    # Helper function to call the API
    def call(self, series_id):
        params = {'series_id': series_id,
                'observation_start': self.start_day,
                'observation_end': self.end_day,
                'api_key': self.api_keys['Fred'],
                'frequency': self.frequency,
                'file_type':'json'}
        response = requests.get(self.metals_url, params = params)
        df = pd.DataFrame(response.json()['observations'])
        return df
    
    def preprocess(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = df['value'].astype(float)
        df = df[['date', 'value']]
        return df
    
    # Helper function to join tables
    def join_tables(self, price, inv, orders):
        price = self.preprocess(price)
        inv = self.preprocess(inv)
        orders = self.preprocess(orders)
        df = price.merge(inv, on='date', how='left', suffixes=('_price', '_inv'))
        df = df.merge(orders, on='date', how='left')
        df.rename(columns={
            'value': 'orders',
            'value_price': 'price',
            'value_inv': 'inventory'
            }, 
                  inplace=True)
        return df
    
    # Helper function to post data
    def post_scrap(self, df, name):
            if not os.path.exists(f'{os.getcwd()}/data/fred'):
                os.makedirs(f'{os.getcwd()}/data/fred')
                print(f"Created directory {os.getcwd()}/data/fred")
            df.to_csv(f'{os.getcwd()}/data/fred/{name}.csv', index=False)
    
    # Fetch ferrous prices
    def fetch_scrap(self):
        ferrous_price = self.call(self.series_ids['ferrous_price'])
        ferrous_inv = self.call(self.series_ids['ferrous_inventory'])
        ferrous_orders = self.call(self.series_ids['ferrous_orders'])
        ferrous = self.join_tables(ferrous_price, ferrous_inv, ferrous_orders)
        self.post_scrap(ferrous, 'ferrous')

        non_ferrous_price = self.call(self.series_ids['non_ferrous_price'])
        non_ferrous_inv = self.call(self.series_ids['alluminum_nonferrous_inventory'])
        non_ferrous_orders = self.call(self.series_ids['non_ferrous_orders'])
        non_ferrous = self.join_tables(non_ferrous_price, non_ferrous_inv, non_ferrous_orders)
        self.post_scrap(non_ferrous, 'non_ferrous')



