from etl.benzinga import Benzinga
from etl.alphavantage import Alphavantage
import datetime
import json
from finbert import FinBert
from etl.target.target import Target
from etl.fred import Fredapi
from etl.yahoofinance import Yahoo
from etl.preprocess.preprocess import Preprocess


class PullData:
    def get_api_keys(self):
        with open("api-keys.json", "r") as f:
            api_keys = json.load(f)
            print(f"Found keys for {', '.join(api_keys.keys())}")
        return api_keys
    
    def update_target(self):
        target = Target()
        target.update()

    def get_data(self, 
                 tick_list, 
                 etfs=None,
                 fromdate="2021-01-01", 
                 todate=datetime.datetime.today().strftime("%Y-%m-%d"),
                 benzinga=True, 
                 yahoo=True, 
                 fred=True,  
                 alpha=True,
                 target=True):
        api_keys = self.get_api_keys()
        fromdate = fromdate
        todate = todate
        if target:
            target = Target()
            target.update()
        if benzinga:
            Benzinga(tick_list, etfs=etfs, api_keys=api_keys).pull_batch_benzinga(fromdate, todate)
            for tick in tick_list:
                try:
                    fin = FinBert(tick)
                    fin.main()
                except Exception as e:
                    print(f"Failed to get sentiment for {tick} | Reason: {e}")
        if yahoo: 
            yahoo = Yahoo(tick_list, etfs, 'daily', api_keys, fromdate, todate)
            yahoo.fetch_data()
            yahoo.export_as_csv()
        if fred:
            fred = Fredapi(api_keys, fromdate, todate)
            fred.fetch_scrap()
        if alpha:
            alpha = Alphavantage(api_keys)
            # for tick in tick_list:
            #     alpha.fetch_earning_data(tick)
            alpha.fetch_sentiment_score()

# working directory should be the same as this file
if __name__ == "__main__":
    etl = PullData()

    fromdate = "2018-01-01"
    todate = datetime.datetime.today().strftime("%Y-%m-%d")
    
    tick_list = [
        'NUE', 'RDUS', 'STLD', 'CTRM', 'LKQ', 'NSC' # stocks
                ]
    etfs = [
        'XME', 'PICK', 'REMX', 'DBB', # metal ETFs
        'GLD', 'IAU', 'GDX', 'GDXJ' # gold ETFs
    ]
    etl.update_target()
    etl.get_data(tick_list, 
                 fromdate=fromdate, 
                 todate=todate, 
                 etfs=etfs,
                 benzinga=False, 
                 yahoo=False, 
                 fred=False, 
                 alpha=False, 
                 target=False)
    
    pipeline = Preprocess(tick_list)
    pipeline.main()