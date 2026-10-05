import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_stock_market_data(tickers, start_date, end_date):
    data = yf.download(
            tickers,
            start="start_date",
            end="end_date",
            interval="1d"
        )

    return

def get_all_tickers():
    df = pd.read_csv('sec_company_tickers.csv')
    return df["ticker"].tolist()

def get_SP_500_tickers():
    df = pd.read_csv('SP500Metadata.csv')
    return df["Symbol"].tolist()

def main():
    # TODO: Read in hurricanes.csv, filter for USA, get stock market data for each ticker around time period of hurricanes
    # df = pd.read_csv('hurricanes.csv')
    # df_usa = df[df['location'].str.contains('USA', na=False)]

    
    print(get_SP_500_tickers())
    get_stock_market_data(get_SP_500_tickers()[0:5], "2026-01-01", "2026-02-01")


    return



main()