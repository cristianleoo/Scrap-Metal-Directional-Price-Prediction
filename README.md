# **Financial News Sentiment Analysis and Directional Price Prediction**

## Overview

This project focuses on analyzing financial news sentiment and utilizing machine learning models to predict stock prices based on the sentiment analysis. It integrates data from various sources, including financial news articles, stock prices, economic indicators, and weather data. The sentiment analysis is performed using two models: FinBert and GPT (Generative Pre-trained Transformer). The machine learning model for price prediction employs the CatBoost algorithm.

Based on the provided project structure and code, it appears to be a comprehensive data pipeline and machine learning project. Here's a description of what the project does:

## Project Structure

The project structure is organized as follows:

```
.
├── LICENSE
├── README.md
├── __pycache__
│   ├── ...
├── api-keys.json
├── catboost_info
│   ├── ...
├── data
│   ├── alphavantage
│   │   ├── alphavantage.csv
│   │   ├── economy_macro.csv
│   │   ├── economy_monetary.csv
│   │   ├── energy_transportation.csv
│   │   └── financial_markets.csv
│   ├── benzinga
│   │   ├── AA.csv
│   │   ├── CLF.csv
│   │   ├── ...
│   ├── finbert
│   │   ├── ...
│   ├── fred
│   │   ├── ferrous.csv
│   │   ├── macro.csv
│   │   └── non_ferrous.csv
│   ├── gmk
│   │   └── gmk.csv
│   ├── gpt
│   │   ├── AA.csv
│   │   ├── CLF.csv
│   │   ├── NSC.csv
│   │   ├── NUE.csv
│   │   └── STLD.csv
│   ├── preprocessed
│   │   └── data.csv
│   ├── ready_to_model
│   │   ├── ...
│   ├── sample
│   │   ├── ...
│   ├── target
│   │   ├── NAT DATABASE.xlsx
│   │   └── target_clean.csv
│   ├── weather
│   │   └── weather.csv
│   └── yahoo
│       └── yahoo.csv
├── etl
│   ├── __pycache__
│   │   ├── ...
│   ├── alphavantage.py
│   ├── benzinga.py
│   ├── benzinga_tool
│   │   ├── ...
│   ├── etl.py
│   ├── fred.py
│   ├── gmk.py
│   ├── preprocess
│   │   ├── __pycache__
│   │   │   └── ...
│   │   └── preprocess.py
│   ├── target
│   │   ├── __pycache__
│   │   │   └── ...
│   │   └── target.py
│   ├── weather.py
│   └── yahoofinance.py
├── fe.png
├── finbert.py
├── gpt.py
├── loss.json
├── main.py
└── models
    ├── __pycache__
    │   ├── ...
    ├── catboost_info
    │   ├── ...
    ├── fengineering.py
    ├── ingest.py
    ├── logs.json
    ├── losses.csv
    ├── losses.json
    ├── mlclassifier.py
    ├── ...
```

## Getting Started

To run the project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/cristian-leo/financial-sentiment-analysis.git
   cd financial-sentiment-analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys:**
   Add your API keys to the `api-keys.json` file.

4. **Run the Main Script:**
   ```bash
   python main.py
   ```

## Data Sources

- **Benzinga:** Financial news articles.
- **GPT:** Sentiment analysis using the Generative Pre-trained Transformer.
- **FinBert:** Sentiment analysis using the FinBert model.
- **Yahoo Finance:** Historical stock prices.
- **FRED:** Federal Reserve Economic Data.
- **AlphaVantage:** Financial market data.
- **Weather:** Weather data.

## **Project Workflow (main.py):**
   - **PullData Class:** A class encapsulating methods for pulling data from various sources.
   - **Update Target:** Updates the target data, potentially serving as a reference or dependent variable for machine learning.
   - **Get Data Method:** Pulls data based on specified flags (e.g., Benzinga, GPT, FinBert, Yahoo, FRED, AlphaVantage, Weather).
   - **Main Execution Block:** Defines the list of tickers, ETFs, and topics, updates the target, pulls data, preprocesses it, and trains the machine learning model.

## Usage

The `main.py` script orchestrates the data pulling process and sentiment analysis. You can customize the tickers, ETFs, topics, and data sources in the script. Additionally, the script includes functionality for updating target data and preprocessing the collected data.

```bash
python main.py
```

## Machine Learning Model

The project employs the CatBoost machine learning algorithm for stock price prediction. The model is trained and evaluated using data from various sources.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The project utilizes pre-trained models such as GPT and FinBert.
- CatBoost, a powerful machine learning library.

## Contributing

Feel free to contribute by opening issues or submitting pull requests.

## Authors

- Cristian Leo

## Contact

For inquiries, please contact [cl4334@columbia.edu].