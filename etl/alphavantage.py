import requests
import csv
from etl.etl import ETL
import os
import pandas as pd
from datetime import datetime

class Alphavantage(ETL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #########################################
    # Helper functions
    def convert_date(self, date_string):
        try:
            #keep everything before the T
            date_string = date_string.split('T')[0]
            dt = datetime.strptime(date_string, '%Y%m%d')
            dt = pd.to_datetime(dt)
            return dt
        except ValueError:
            return date_string
        
    def range_dates(self, start_date:str):
        date_range = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq='D')
        date_list = date_range.strftime('%Y-%m-%d').tolist()
        return date_list
    
    def next_date(self, date):
        date = pd.to_datetime(date)
        return (date + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    
    def get_last_date(self):
        df = pd.read_csv(f'{os.getcwd()}/data/alphavantage.csv')
        date = df['time_published'].max()
        date = pd.to_datetime(date)
        date = date.strftime('%Y-%m-%d')
        return date
    
    #########################################
    # Earnings
    def fetch_earning_data(self, ticker:str):
        # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={self.api_keys["Alphavantage"]}'
        r = requests.get(url)
        data = r.json()

        # Check if the 'quarterlyEarnings' key exists in the response
        if 'quarterlyEarnings' in data:
            earnings = data['quarterlyEarnings']

            # Specify the output CSV file path
            # create the data folder if it doesn't exist
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
    # Sentiment
    def fetch_sentiment_score(self, start_day:str='2018-01-01', erase=False): #YYYYMMDD format
        if not erase:
            try:
                start_day = self.get_last_date()
            except Exception:
                pass
        dates = self.range_dates(start_day)
        df = pd.DataFrame()
        for date in dates:
            try:
                #next_date = self.next_date(date)
                url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={date.replace("-", "")}T0130&sort=RELEVANCE&apikey={self.api_keys["Alphavantage"]}'
                r = requests.get(url)
                data = r.json()
                extracted_data = []

                # Loop through each entry in the 'feed'
                for entry in data['feed']:
                    # Check if 'ticker_sentiment' is not empty
                    if entry['ticker_sentiment']:
                        # Extract the desired information
                        extracted_info = {
                            'time_published': entry['time_published'],
                            'relevance_score': entry['ticker_sentiment'][0]['relevance_score'],
                            'ticker': entry['ticker_sentiment'][0]['ticker'],
                            'ticker_sentiment_label': entry['ticker_sentiment'][0]['ticker_sentiment_label'],
                            'ticker_sentiment_score': entry['ticker_sentiment'][0]['ticker_sentiment_score']
                        }
                    else:
                        # If 'ticker_sentiment' is empty, set all fields to None
                        extracted_info = {
                            'time_published': None,
                            'relevance_score': None,
                            'ticker': None,
                            'ticker_sentiment_label': None,
                            'ticker_sentiment_score': None
                        }

                    # Append the extracted information to the list
                    extracted_data.append(extracted_info)

                    # Create a DataFrame from the extracted data
                    new_df = pd.DataFrame(extracted_data)
                    new_df.dropna(how='all', inplace=True)
                    new_df['time_published'] = new_df['time_published'].apply(self.convert_date)
                    pd.concat([df, new_df], ignore_index=True)
            except Exception as e:
                print(f"Failed to get sentiment for {date} | Reason: {e}")
                break

        output_file = f'{os.getcwd()}/data/alphavantage.csv' 
        # Open the output file in write mode
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(df.columns)

            # Write the data rows
            for _, row in df.iterrows():
                csv_writer.writerow(row)

        print(f"Data has been successfully converted and saved to '{output_file}'.")