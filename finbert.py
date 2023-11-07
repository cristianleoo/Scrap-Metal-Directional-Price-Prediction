from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import time
import os

class FinBert:
    def __init__(self, tickr, max_batch=500, max_length=256, load=False, input='title'):
        self.tickr = tickr
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
        try:
            df = pd.read_csv(f"{os.getcwd()}/finbert/{self.tickr}.csv")
        except Exception:
            df = pd.read_csv(f"{os.getcwd()}/data/finbert/{self.tickr}.csv")
        return df
    
    def get_last_date(self):
        df = self.get_sentiment()
        return df['created'].iloc[-1]

    def get_benzinga(self):
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
        inputs = self.tokenizer(df[self.input][start_batch:end_batch].to_list(), padding=True, truncation=True, max_length=256, return_tensors="pt")
        outputs = self.model(**inputs).logits
        return outputs
    
    def process_outputs(self, outputs):
        outputs = outputs.argmax(1).numpy()
        df = self.get_benzinga()
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

    
    # move for loop outside of the function
    def main(self):
        df = self.get_benzinga()
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
    
    