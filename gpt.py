import numpy as np
import pandas as pd
import time
import os
from openai import OpenAI
from timeout_decorator import timeout, TimeoutError
import json

class GPT:
    def __init__(self, tickr, df=None, input='body', timeout=30, load=True):
        """
        Initializes the GPT class.

        Parameters:
        - tickr (str): The ticker symbol of the stock.
        - df (pandas.DataFrame, optional): The DataFrame containing the data. Default is None.
        - input (str, optional): The column name of the input data. Default is 'body'.
        - timeout (int, optional): The timeout value for API requests. Default is 30.
        - load (bool, optional): Whether to load the sentiment data or not. Default is True.
        """
        with open('api-keys.json') as f:
            api_keys = json.load(f)
        
        self.client = OpenAI(api_key=api_keys['openai'])
        self.tickr = tickr
        self.df = df
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
        self.load = load
        self.input = input
        self.timeout = timeout

    def get_sentiment(self):
        """
        Retrieves the sentiment data for the given ticker symbol.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the sentiment data.
        """
        df = pd.read_csv(f"{self.data_path}/gpt/{self.tickr}.csv")
        return df
    
    def get_last_date(self):
        """
        Retrieves the last date of the sentiment data.

        Returns:
        - last_date (str): The last date of the sentiment data.
        """
        df = self.get_sentiment()
        last_date = df['created'].iloc[-1]
        return last_date

    def get_benzinga(self):
        """
        Retrieves the Benzinga data.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the Benzinga data.
        """
        if self.load:
            sentiment = self.get_sentiment()
            df = sentiment.dropna(subset=[self.input])
        else:
            df = pd.read_csv(f"{self.data_path}/benzinga/{self.tickr}.csv")
        df = df.reset_index(drop=True)
        return df
    
    def check_score(self, score):
        """
        Checks if the given score is valid.

        Parameters:
        - score (int): The score to be checked.

        Returns:
        - valid (bool): True if the score is valid, False otherwise.
        """
        try:
            score = int(score)
            if score < 0 or score > 2:
                return False
            else:
                return True
        except Exception:
            return False
    
    def score_article(self, ticker, article):
        """
        Scores the given article.

        Parameters:
        - ticker (str): The ticker symbol of the stock.
        - article (str): The article to be scored.

        Returns:
        - score (int or np.nan): The score of the article, or np.nan if scoring fails.
        """
        valid = False
        if len(article) > 4096:
            article = article[:4096]
        errors = 0
        while not valid and errors < 5:
            try:
                @timeout(self.timeout)
                def completion_request():
                    return self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        max_tokens=1,
                        messages=[
                            {"role": "system", "content": f"You are a helpful market sentiment tool. You provide a score from 0 to 2 for the sentiment score of the news in relation to the stock {ticker}, where 0 is negative sentiment, 1 is neutral sentiment, and 2 is positive sentiment. Return only the score."},
                            {"role": "user", "content": article}
                        ]
                    )

                completion = completion_request()
                score = completion.choices[0].message.content
                valid = self.check_score(score)
                if not valid:
                    print(f"Invalid score: {score} | Retrying...")
            except TimeoutError:
                print("Timeout occurred. Skipping article...")
                score = np.nan
                break
            except Exception as e:
                print(f"Error occurred: {e} | Retrying...")
            finally:
                errors += 1
        if errors >= 5:
            score = np.nan
            print(f"Skipping article because of too many errors...")
        return score
    
    def main(self):
        """
        Performs the main sentiment analysis process.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the sentiment data.
        """
        print(f"Getting sentiment for {self.tickr} using GPT-3.5-turbo")
        df = self.get_benzinga()
        scores = []
        start = time.time()
        start_batch = time.time()
        checkpoint = 100  # Set the checkpoint value to 100
        checkpoint_counter = 0  # Counter to keep track of the number of iterations
        if 'sentiment' not in df.columns:
            df['sentiment'] = np.nan
            start_idx = 0
        else:
            start_idx = df['sentiment'].last_valid_index() + 1

        if start_idx >= len(df):
            print(f"Sentiment data for {self.tickr} is already up to date")
            return df
        else:
            print("*************************************")
            print(f"Starting sentiment analysis for {self.tickr} at index {start_idx} | {len(df[start_idx:])} articles")
            for i in range(len(df[start_idx:])):
                scores.append(self.score_article(self.tickr, df[self.input][i+start_idx]))
                if i % 10 == 0 and i != 0:
                    time_batch = time.time()
                    print(f"Completed {i} articles in {time_batch - start_batch:.1f} seconds | {i/len(df):.2%} complete")
                    start_batch = time.time()
                if i % checkpoint == 0 and i != 0:
                    df['sentiment'].iloc[checkpoint_counter * checkpoint+start_idx:i+start_idx] = scores[checkpoint_counter * checkpoint:i]
                    checkpoint_counter += 1
                    df.to_csv(f"{self.data_path}/gpt/{self.tickr}.csv", index=False)
                    print(f"Saved sentiment data for {self.tickr} at checkpoint {start_idx +i} | Time taken: {time.time() - start:.1f} seconds")

            df['sentiment'].iloc[checkpoint_counter * checkpoint + start_idx:] = scores[checkpoint_counter * checkpoint:]
            end = time.time()
            df['created'] = pd.to_datetime(df['created']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df.dropna(subset=['sentiment'], inplace=True)
            if not os.path.exists(f'{self.data_path}/gpt'):
                os.makedirs(f'{self.data_path}/gpt')
                print(f"Created directory {self.data_path}/gpt")
            df.to_csv(f"{self.data_path}/gpt/{self.tickr}.csv", index=False)
            print(f"Saved sentiment data for {self.tickr} | Time taken: {(end - start)/60:.1f} minutes")
            return df