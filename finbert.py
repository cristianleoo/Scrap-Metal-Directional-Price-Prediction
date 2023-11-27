from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import time
import os

class FinBert:
    def __init__(self, tickr, df=None, max_batch=500, max_length=256, load=False, input='title'):
        """
        Initialize the FinBert class.

        Parameters:
        - tickr (str): The ticker symbol of the stock.
        - df (pandas.DataFrame, optional): The DataFrame containing the data to be processed. Default is None.
        - max_batch (int, optional): The maximum number of data points to process in each batch. Default is 500.
        - max_length (int, optional): The maximum length of the input sequence. Default is 256.
        - load (bool, optional): Whether to load existing sentiment data. Default is False.
        - input (str, optional): The column name of the input data. Default is 'title'.
        """
        self.tickr = tickr
        self.df = df
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.max_batch = max_batch
        self.max_length = max_length
        self.load = load
        self.load_passed = False
        self.incomplete_predictions = False
        self.input = input
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("GPU is available and being used")
        else:
            try:
                self.device = torch.device("mps")
                print("GPU is not available, using MPS instead")
            except Exception:
                self.device = torch.device("cpu")
                print("GPU is not available, using CPU instead")

    def get_sentiment(self):
        """
        Get the sentiment data for the stock.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the sentiment data.
        """
        try:
            df = pd.read_csv(f"{os.getcwd()}/finbert/{self.tickr}.csv")
        except Exception:
            df = pd.read_csv(f"{os.getcwd()}/data/finbert/{self.tickr}.csv")
        return df
    
    def get_last_date(self):
        """
        Get the last date of the sentiment data.

        Returns:
        - last_date (str): The last date of the sentiment data.
        """
        df = self.get_sentiment()
        return df['created'].iloc[-1]

    def get_benzinga(self):
        """
        Get the data from Benzinga.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the data from Benzinga.
        """
        path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(f"{path}/data/benzinga/{self.tickr}.csv")
        df = df[['created', self.input]]
        df = df.dropna()
        df = df.reset_index(drop=True)

        if self.load:
            try:
                df = df[df['created']>=self.get_last_date()]
                self.load_passed = True
            except Exception as e:
                print(e)
                pass

        return df
    
    def predict_sentiment(self, df, start_batch, end_batch):
        """
        Predict the sentiment of the input data.

        Parameters:
        - df (pandas.DataFrame): The DataFrame containing the input data.
        - start_batch (int): The starting index of the batch.
        - end_batch (int): The ending index of the batch.

        Returns:
        - outputs (torch.Tensor): The predicted sentiment scores.
        """
        inputs = self.tokenizer(df[self.input][start_batch:end_batch].to_list(), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        outputs = self.model(**inputs).logits
        return outputs
    
    def process_outputs(self, outputs):
        """
        Process the predicted sentiment scores.

        Parameters:
        - outputs (torch.Tensor): The predicted sentiment scores.

        Returns:
        - df (pandas.DataFrame): The DataFrame containing the processed sentiment data.
        """
        outputs = outputs.argmax(1).numpy()
        if self.df is None:
            df = self.get_benzinga()
        else:
            df = self.df
        if len(outputs) < len(df):
            self.incomplete_predictions = True
            print(f"Warning: incomplete predictions for {self.tickr}")
            print(f"Number of predictions: {len(outputs)}")
            print(f"Number of {self.input}s: {len(df)}")
            print("Please run it again to get the complete predictions")

        df = df[:len(outputs)]
        df['sentiment'] = outputs
        return df

    def save_sentiment(self, predicted_sentiment):
        """
        Save the predicted sentiment data.

        Parameters:
        - predicted_sentiment (pandas.DataFrame): The DataFrame containing the predicted sentiment data.
        """
        if self.df is None:
            self.get_benzinga()
        if self.load_passed:
            print(f"Updating new data from {self.get_last_date()}")
            sentiment = self.get_sentiment()
            sentiment_new = predicted_sentiment
            sentiment = pd.concat([sentiment, sentiment_new])
            sentiment = sentiment.sort_values(by='created')
            sentiment = sentiment.drop_duplicates(subset=['created'], keep='first')
            sentiment = sentiment[['created', 'sentiment']]
            if not os.path.exists(f'{os.getcwd()}/data/finbert'):
                os.makedirs(f'{os.getcwd()}/data/finbert')
                print(f"Created directory {os.getcwd()}/data/finbert")
            sentiment.to_csv(f"{os.getcwd()}/data/finbert/{self.tickr}.csv", index=False)
            print(f"Saved sentiment data for {self.tickr}")

        else:
            sentiment = predicted_sentiment
            if not os.path.exists(f'{os.getcwd()}/data/finbert'):
                os.makedirs(f'{os.getcwd()}/data/finbert')
                print(f"Created directory {os.getcwd()}/data/finbert")
            sentiment.to_csv(f"{os.getcwd()}/data/finbert/{self.tickr}.csv", index=False)
            print(f"Saved sentiment data for {self.tickr}")

    
    def main(self):
        """
        Main function to process the data and predict sentiment.

        Returns:
        - None
        """
        if self.df is None:
            df = self.get_benzinga()
        else:
            df = self.df
        n = len(df[self.input])//self.max_batch
        print(f"Starting encoding {n+1} batches of data for {self.tickr} | N. {self.input}s: {len(df[self.input])}")
        if n == 0:
            outputs = self.process_outputs(torch.cat([self.predict_sentiment(df, 0, -1)], 0))
            self.save_sentiment(outputs)
        else:
            for i in np.arange(0, n+1):
                if (i+1)*self.max_batch > len(df[self.input]):
                    outputs = self.process_outputs(torch.cat([self.predict_sentiment(df, i*self.max_batch+1, -1)], 0))
                    self.save_sentiment(outputs)
                else:
                    outputs = self.process_outputs(torch.cat([self.predict_sentiment(df, i*self.max_batch+1, (i+1)*self.max_batch+1)]))
                    self.save_sentiment(outputs)
                print(f"Finished encoding batch {i+1}")
                time.sleep(1)

        print("Finished encoding all batches")
        #return torch.cat(outputs, 0)
