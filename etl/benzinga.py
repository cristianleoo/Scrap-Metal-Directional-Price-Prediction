from etl.etl import ETL
from etl.benzinga_tool import financial_data, news_data
from etl.benzinga_tool.news_data import News
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import datetime
import os

# Class for extracting, transforming, and loading data from Benzinga API
class Benzinga(ETL):
    def __init__(self, tick_list, etfs=None, display_output='full', *args, **kwargs):
        """
        Initialize the Benzinga class.

        Parameters:
        - tick_list: list of ticker symbols to retrieve data for
        - etfs: list of ETF symbols to include in the tick_list
        - display_output: display option for news data (abstract, full, or headline)
        - *args, **kwargs: additional arguments and keyword arguments for the parent class
        """
        super().__init__(*args, **kwargs)
        self.tick_list = tick_list
        self.etfs = etfs
        if self.etfs:
            self.tick_list.extend(self.etfs)
        self.start_day = None
        self.end_day = None
        self.display_output = display_output
        self.retrieving_all = False

    def clean_text(self, text):
        """
        Clean the text by removing HTML tags and special characters.

        Parameters:
        - text: input text to be cleaned

        Returns:
        - clean_text: cleaned text
        """
        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text().replace('\n', ' ').replace('"', '').replace('\u2028', ' ').replace('\u2029', ' ').replace('\r', ' ')
        return ' '.join(clean_text.split())

    def process_stories(self, stories, ticker):
        """
        Process the retrieved stories by cleaning the data and filtering based on ticker.

        Parameters:
        - stories: list of stories to be processed
        - ticker: ticker symbol to filter the stories

        Returns:
        - df: processed DataFrame containing the relevant stories
        """
        if self.retrieving_all:
            df = pd.DataFrame(stories)
            df['stocks'] = df['stocks'].apply(lambda x: [entry['name'] for entry in x])
            df['stocks'] = df['stocks'].apply(lambda x: ', '.join(x))
            df['stocks'] = df['stocks'].apply(str.upper)
            try:
                df = df[df['stocks'].str.contains(ticker)][['created', 'title', 'body']]
                print(df.head())
                print(f"Found {len(df)} stories for {ticker}")
            except Exception:
                print(f"No stories found for {ticker}")
                df = None
        else:
            df = pd.DataFrame(stories)[['created', 'title', 'body']]
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
        df['title'] = df['title'].apply(lambda x: x.replace('"', ''))
        df['body'] = df['body'].apply(self.clean_text)
        df = df.dropna(subset=['created', 'title', 'body'])
        df['created'] = df['created'].dt.date
        return df

    def benzinga_call(self, news, ticker, fromdate, todate):
        """
        Make requests to the Benzinga API to retrieve news data.

        Parameters:
        - news: Benzinga News object
        - ticker: ticker symbol to retrieve data for
        - fromdate: start date for the data retrieval
        - todate: end date for the data retrieval

        Returns:
        - df: processed DataFrame containing the retrieved news data
        """
        stories = news.news(display_output=self.display_output, company_tickers=ticker, pagesize=100, date_from=fromdate, date_to=todate)
        print(f"A. Found {len(stories)} stories in the response.")
        if len(stories) == 0:
            self.retrieving_all = True
            print("Stories for single ticker not found. Retrieving all stories.")
            stories = news.news(display_output=self.display_output, pagesize=100, date_from=fromdate, date_to=todate)
        else:
            if len(pd.DataFrame(stories)) == 0:
                self.retrieving_all = True
                print("Stories for single ticker not found. Retrieving all stories.")
                stories = news.news(display_output=self.display_output, pagesize=100, date_from=fromdate, date_to=todate)
            else:
                print(f"B. Found {len(stories)} stories in the response.")
        
        df = self.process_stories(stories, ticker=ticker)

        if df is not None:
            fromdate = (df.iloc[-1, 0] - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            one_month_be4_todate = (datetime.datetime.strptime(todate, '%Y-%m-%d') - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            last_request_fromdate = None
            while fromdate < one_month_be4_todate:
                if last_request_fromdate is not None and fromdate <= last_request_fromdate:
                    fromdate = (datetime.datetime.strptime(last_request_fromdate, '%Y-%m-%d') + datetime.timedelta(days=15)).strftime('%Y-%m-%d')
                    continue

                if self.retrieving_all:
                    stories = news.news(display_output=self.display_output, pagesize=100, date_from=fromdate, date_to=todate)
                else:
                    stories = news.news(display_output=self.display_output, company_tickers=ticker, pagesize=100, date_from=fromdate, date_to=todate)
                stories_df = self.process_stories(stories, ticker=ticker)
                df = pd.concat([df, stories_df]).drop_duplicates(subset=['title']).reset_index(drop=True)
                last_request_fromdate = fromdate
                fromdate = (df.iloc[-1, 0] - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        return df

    def get(self, tick):
        """
        Retrieve news data for a single ticker.

        Parameters:
        - tick: ticker symbol to retrieve data for

        Returns:
        - df: processed DataFrame containing the retrieved news data
        """
        news = News(api_token=self.api_keys['Benzinga'])
        df = self.benzinga_call(news, ticker=tick, fromdate=self.start_day, todate=self.end_day)
        if df is not None:
            print(f"Found {len(df)} stories for {tick}")
            return df
        else:
            df = self.benzinga_call(news, ticker=None, fromdate=self.start_day, todate=self.end_day)
    
    def pull_batch_benzinga(self, start_day, end_day):
        """
        Batch retrieve news data for multiple tickers.

        Parameters:
        - start_day: start date for the data retrieval
        - end_day: end date for the data retrieval
        """
        self.start_day = start_day
        self.end_day = end_day

        for tick in self.tick_list:
            try:
                df = self.get(tick)
                if df is not None:
                    df = df.sort_values(by='created')
                    df = df.drop_duplicates(subset=['created', 'title'], keep='first')
                    if not os.path.exists(f'{os.getcwd()}/data/benzinga'):
                        os.makedirs(f'{os.getcwd()}/data/benzinga')
                        print(f"Created directory {os.getcwd()}/data/benzinga")
                    df.to_csv(f'{os.getcwd()}/data/benzinga/{tick}.csv', index=False)
            except Exception as e:
                print(f"Failed to get data for {tick} | Reason: {e}")
                continue