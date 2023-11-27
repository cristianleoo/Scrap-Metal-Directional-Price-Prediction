import requests
import csv
from etl.etl import ETL
import os
import pandas as pd
from datetime import datetime

class Alphavantage(ETL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This class extends the ETL class and provides methods for fetching earnings and sentiment data from the Alphavantage API,
        # converting and saving the data, and performing other helper functions.

        class Alphavantage(ETL):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            #########################################
            # Helper functions

            # This function converts a date string in the format 'YYYYMMDD' to a pandas datetime object.
            def convert_date(self, date_string):
                try:
                    # Keep everything before the 'T' character
                    date_string = date_string.split('T')[0]
                    dt = datetime.strptime(date_string, '%Y%m%d')
                    dt = pd.to_datetime(dt)
                    return dt
                except ValueError:
                    return date_string

            # This function generates a list of dates starting from the specified start_date up to the current date,
            # with a frequency of one month.
            def range_dates(self, start_date:str):
                date_range = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq='M')
                date_list = date_range.strftime('%Y-%m-%d').tolist()
                return date_list

            # This function calculates the next date given a date string in the format 'YYYY-MM-DD'.
            def next_date(self, date):
                date = pd.to_datetime(date)
                return (date + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

            # This function retrieves the last date from the 'time_published' column of the alphavantage.csv file.
            def get_last_date(self):
                df = pd.read_csv(f'{os.getcwd()}/data/alphavantage.csv')
                date = df['time_published'].max()
                date = pd.to_datetime(date)
                date = date.strftime('%Y-%m-%d')
                return date

            #########################################
            # Earnings

            # This function fetches earnings data for a given ticker from the Alphavantage API,
            # converts the data to CSV format, and saves it to a file.
            def fetch_earning_data(self, ticker:str):
                # Replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
                url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={self.api_keys["Alphavantage"]}'
                r = requests.get(url)
                data = r.json()

                # Check if the 'quarterlyEarnings' key exists in the response
                if 'quarterlyEarnings' in data:
                    earnings = data['quarterlyEarnings']

                    # Specify the output CSV file path
                    # Create the data folder if it doesn't exist
                    if not os.path.exists(f'{os.getcwd()}/data/{ticker}'):
                        os.makedirs(f'{os.getcwd()}/data/{ticker}')
                    output_file = f'{os.getcwd()}/data/{ticker}/earnings.csv' 

                    # Open the output file in write mode
                    with open(output_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        # Write the header row
                        writer.writerow(earnings[0].keys())

                        # Write the data rows
                        for earnings_data in earnings:
                            writer.writerow(earnings_data.values())

                    print(f"Data has been successfully converted and saved to '{output_file}'.")
                else:
                    print("No earnings data found in the response.")

            #########################################
            # Save data

            # This function saves a pandas DataFrame to a CSV file.
            def save_data(self, df, ticker=None, topic=None):
                path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/alphavantage')
                if not os.path.exists(path):
                    os.makedirs(path)
                    print(f"Created directory {path}")
                if ticker:
                    output_file = f'{path}/{ticker}.csv'
                elif topic:
                    output_file = f'{path}/{topic}.csv'
                else:
                    output_file = f'{path}/alphavantage.csv'
                    
                # Open the output file in write mode
                with open(output_file, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(df.columns)

    #########################################
    # Main
    def main(self, 
             tick_list=None, 
             topic_list=None, 
             erase=False
             ):
        if tick_list:
            for tick in tick_list:
                # self.fetch_earning_data(tick)
                self.fetch_sentiment_score(ticker=tick, erase=erase)
        if topic_list:
            for topic in topic_list:
                self.fetch_sentiment_score(topic=topic, erase=erase)
        if not tick_list and not topic_list:
            self.fetch_sentiment_score(erase=erase)
        