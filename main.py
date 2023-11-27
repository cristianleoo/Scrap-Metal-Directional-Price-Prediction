# Importing necessary modules
from etl.benzinga import Benzinga
from etl.alphavantage import Alphavantage
import datetime
import json
from finbert import FinBert
from gpt import GPT
from etl.target.target import Target
from etl.fred import Fredapi
from etl.yahoofinance import Yahoo
from etl.weather import WeatherScraper
from etl.preprocess.preprocess import Preprocess
from models.ingest import Ingest
from models.mlclassifier import Trainer

# Class for pulling data from various sources
class PullData:
    """
    Class to pull data from various sources.
    """

    def get_api_keys(self):
        """
        Method to get API keys from a JSON file.

        Returns:
            dict: A dictionary containing the API keys.
        """
        with open("api-keys.json", "r") as f:
            api_keys = json.load(f)
            print(f"Found keys for {', '.join(api_keys.keys())}")
        return api_keys
    
    def update_target(self):
        """
        Method to update the target data.
        """
        target = Target()
        target.update()

    def get_data(self, 
                 tick_list, 
                 etfs=None,
                 topics=None,
                 fromdate="2000-01-01", 
                 todate=datetime.datetime.today().strftime("%Y-%m-%d"),
                 benzinga=True,
                 gpt=True,
                 finbert=True,
                 yahoo=True, 
                 fred=True,  
                 alpha=True,
                 weather=True,
                 target=True):
        """
        Method to get data from different sources.

        Args:
            - tick_list (list): List of tickers.
            - etfs (list, optional): List of ETFs. Defaults to None.
            - topics (list, optional): List of topics. Defaults to None.
            - fromdate (str, optional): Start date. Defaults to "2000-01-01".
            - todate (str, optional): End date. Defaults to current date.
            - benzinga (bool, optional): Flag to pull data from Benzinga. Defaults to True.
            - gpt (bool, optional): Flag to pull data using GPT. Defaults to True.
            - finbert (bool, optional): Flag to pull data using FinBert. Defaults to True.
            - yahoo (bool, optional): Flag to pull data from Yahoo. Defaults to True.
            - fred (bool, optional): Flag to pull data from FRED. Defaults to True.
            - alpha (bool, optional): Flag to pull data from AlphaVantage. Defaults to True.
            - weather (bool, optional): Flag to pull weather data. Defaults to True.
            - target (bool, optional): Flag to update target data. Defaults to True.
        """

        api_keys = self.get_api_keys()
        fromdate = fromdate
        todate = todate

        if target:
            target = Target()
            target.update()

        if benzinga:
            Benzinga(tick_list, etfs=etfs, api_keys=api_keys).pull_batch_benzinga(fromdate, todate)

        if gpt:
            if etfs:
                benzinga_ticks = tick_list + etfs
            for tick in ['NUE','STLD','NSC', 'CLF', 'AA']: #benzinga_ticks:
                try:
                    gpt = GPT(tick,
                                input='body',
                                timeout=30,
                                load=True
                            )
                    gpt.main()
                except Exception as e:
                    print(f"Failed to get sentiment for {tick} | Reason: {e}")

        if finbert:
            if etfs:
                benzinga_ticks = tick_list + etfs
            for tick in benzinga_ticks:
                try:
                    fin = FinBert(tick, 
                                  load=True,
                                  input='body',
                                  max_batch=50)
                    fin.main()
                except Exception as e:
                    print(f"Failed to get sentiment for {tick} | Reason: {e}")

        if yahoo:
            yahoo = Yahoo(tick_list, etfs, 'monthly', api_keys, fromdate, todate)
            yahoo.fetch_data()
            yahoo.export_as_csv()

        if fred:
            fred = Fredapi(api_keys, fromdate, todate)
            fred.fetch_scrap()
            fred.fetch_macro_data()

        if alpha:
            alpha = Alphavantage(api_keys)
            alpha.main(topic_list=topics)

        if weather:
            weather = WeatherScraper(api_keys)
            weather.scrape_weather_data()

# Main execution block
if __name__ == "__main__":
    etl = PullData()

    # Set the date range for data retrieval
    fromdate = "2000-01-01"
    todate = datetime.datetime.today().strftime("%Y-%m-%d")
    
    # Define the list of tickers and ETFs
    tick_list = [
        'NUE',
        'STLD',
        'LKQ',
        'NSC',
        'CLF',
        'AA',
        'VMC',
        'MLM',
        'TXI'
    ]

    etfs = [
        'XME',
        'PICK',
        'REMX',
        'DBB',
        'GLD',
        'IAU',
        'GDX',
        'GDXJ',
        'IYM',
        'XLB',
        'VAW',
        'FXZ',
        'DJP',
        'IYT'
    ]

    topics = [
        'financial_markets',
        'economy_monetary',
        'economy_macro',
        'energy_transportation'
    ]

    # Update the target data
    etl.update_target()

    # Get data from various sources
    etl.get_data(tick_list, 
                 fromdate=fromdate, 
                 todate=todate, 
                 etfs=etfs,
                 topics=topics,
                 benzinga=False, 
                 finbert=False,
                 gpt=False,
                 yahoo=False, 
                 fred=False, 
                 alpha=False, 
                 weather=False,
                 target=False)
    
    # Preprocess the data
    pipeline = Preprocess(tick_list)
    pipeline.main(
        benzinga=False, # if benzinga=True it will return a dataset with the benzinga articles instead of the finbert sentiment
        sentiment_model='gpt', # 'finbert' or 'gpt'
    )

    ingest = Ingest()
    X_train, X_val, X_test, y_train, y_val, y_test = ingest.split(val=False,
                                                                drop_unimportant=False, 
                                                                transform=True,
                                                                #    new_cols=new_cols,
                                                                scale=False
                                                                ) # drop_unimportant=False give the best result for CatBoost
    
    trainer = Trainer()
    test_preds, losses = trainer.main(X_train, X_test, y_train, y_test)
    losses