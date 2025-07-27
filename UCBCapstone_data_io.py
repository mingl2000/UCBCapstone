import yfinance as yf
import pandas as pd
import numpy as np
import os
from pypinyin import pinyin, Style
'''
    This module provides functions to read and process financial data related to futures trading in China.
    Please note some of the data has been downloaded from CFFEX (China Financial Futures Exchange) and preprocessed for analysis.

'''

''' 
    Data downloaded from CFFEX http://www.cffex.com.cn and preprocessed for analysis.
'''
def read_CSI_future_index_history(product='IF'):
    df= pd.read_csv(f"data/cffex_{product}.csv", encoding="gb2312",index_col='datetime', parse_dates=True, date_format='%Y%m%d')
    df.columns=['Contract','Open','High','Low','Volume','Amount','OpenInterest', 'OpenInterestChange', 'Close', 'SettlementPrice','LastSettlementPrice','Change1','Change2','Delta']
    return df


def read_CSI_future_trading_data(product='IF'):

    df =pd.read_csv(f'data/future_trading_{product}.csv', encoding="gb18030", index_col='datetime',parse_dates=True, date_format='%Y%m%d')
    
    df=df.rename(columns={'Unnamed: 0':'dealer'})
    df['dealer']= df['dealer'].apply(lambda x:  ''.join(word[0] for word in pinyin(x, style=Style.NORMAL)) )

    #df_future['dealer']=df_future['dealer']
    if product=='IF':
        df=df[df.index >= '2021-03-11']

    return df


def read_yahoo_data(ticker, start_date, end_date, cache_dir='data'):
    """
    Read historical data from Yahoo Finance for a given ticker and date range.
    """
    data_file=f'{cache_dir}/{ticker}.csv'
    if os.path.exists(data_file):
        df = pd.read_csv(data_file,  index_col=0,parse_dates=True, date_format='%Y-%m-%d')
        print(f'{ticker} already exists, skip downloading...')
    else:
        df =yf.download(ticker)
        df = df.droplevel('Ticker',axis=1)
        df.to_csv(data_file)

    return df[(df.index>=start_date) & (df.index<=end_date)]

def read_all(product='IF'):
    idx_future_map={
        'IF':'000300.SS', #CSI 300
        'IH':'510050.SS',  #CSI 50
        'IC':'510500.SS',  #ZZ 500
        'IM':'512100.SS'   #ZZ 1000

    }
    df_index_future=read_CSI_future_index_history(product)
    start_date=min(df_index_future.index)
    last_date=max(df_index_future.index)
    df_index = read_yahoo_data(idx_future_map[product], start_date, last_date)
    df_future_trading=read_CSI_future_trading_data(product)


    return df_index_future, df_future_trading, df_index

if __name__ == "__main__":
    product = 'IF'
    df_index_future, df_future_trading, df_index =read_all(product='IF')
    print(f"Future Trading Data for {product}: df_index_future: {len(df_index_future)} rows  df_future_trading: {len(df_future_trading)} rows df_index:  {len(df_index)} rows " )
