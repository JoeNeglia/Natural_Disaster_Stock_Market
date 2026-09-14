import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_stock_market_data(start_date, end_date):
    # data = yf.download(
    #     tickers,
    #     start="2020-01-01",
    #     end="2026-01-01",
    #     interval="1d"
    # )

    return

def get_tickers():
    df = pd.read_csv('sec_company_tickers.csv')
    return df["ticker"].tolist()

def main():
    # TODO: Read in hurricanes.csv, filter for USA, get stock market data for each ticker around time period of hurricanes
    # df = pd.read_csv('hurricanes.csv')
    # df_usa = df[df['location'].str.contains('USA', na=False)]

    
    print(get_tickers())




    return



main()