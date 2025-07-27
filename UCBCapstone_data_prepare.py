import pandas as pd
import numpy as np
import os
from UCBCapstone_data_io import read_yahoo_data
'''
    Add technical indicators to the financial data.
'''

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    Parameters:
        data: pandas Series or DataFrame with 'Close' prices
        period: lookback period for RSI calculation (default: 14)
    Returns:
        pandas Series with RSI values
    """
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses over the period
    avg_gains = gains.rolling(window=period, min_periods=1).mean()
    avg_losses = losses.rolling(window=period, min_periods=1).mean()
    
    # Calculate Relative Strength (RS)
    rs = avg_gains / (avg_losses + 1e-10)  # Add small constant to avoid division by zero
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.rename('RSI')
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD, Signal Line, and Histogram using pandas.
    Parameters:
        data: pandas Series or DataFrame with 'Close' prices
        fast_period: period for fast EMA (default: 12)
        slow_period: period for slow EMA (default: 26)
        signal_period: period for signal line EMA (default: 9)
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    # Ensure data is a Series
    if isinstance(data, pd.DataFrame):
        data = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
    
    # Calculate Fast and Slow EMAs
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD Line
    macd = fast_ema - slow_ema
    
    # Calculate Signal Line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd - signal
    
    # Return DataFrame
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal,
        'Histogram': histogram
    }, index=data.index)

def add_technical_indicators(df):
    """
    Add technical indicators to the futures trading data.
    """
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df_macd= calculate_macd(df["Close"])

    df['MACD'] = df_macd['MACD']    
    df['Signal'] = df_macd['Signal']    
    df['Histogram'] = df_macd['Histogram']    
    df['Bollinger_High'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Low'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    
    return df


def add_technical_return(df):
    """
    Add technical return indicators to the futures trading data.
    """
    df['Return'] = df['Close'].pct_change()
    #df['Cumulative_Return'] = (1 + df['Return']).cumprod() - 1
    #df['Volatility'] = df['Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
    
    return df


def group_future_trading_by_date(df_trading):
    future_cols=df_trading.columns.drop(['dealer'])
    future_cols_without_dealer=df_trading.columns.drop(['dealer'])
    df_future_by_date=df_trading[future_cols_without_dealer].groupby(df_trading.index).sum()
    return df_future_by_date



def pivote_df_future(df_future_top):
    df_future_pivots=[]
    #for col in df_future_top.columns.drop(['dealer','datetime']):
    for col in df_future_top.columns.drop(['dealer']):
        df_future_pivot=df_future_top.pivot(columns='dealer', values=col)
        df_future_pivot=df_future_pivot.fillna(0)
        df_future_pivot.columns= [x +f'_{col}' for x in df_future_pivot.columns]
        df_future_pivots.append(df_future_pivot)

    df_future_pivoted = pd.concat(df_future_pivots, axis=1)
    return df_future_pivoted
    


def get_idx_future_top_traders(df_future, n):
    df_trader_vol= pd.DataFrame(df_future.groupby('dealer')['volume'].sum())
    df_trader_vol = df_trader_vol.sort_values(by='volume', ascending=False)
    return df_future[ df_future['dealer'].isin(df_trader_vol.index[:n])]

    df_trader_vol['volume'][n]


def get_total_trade_vol_top_dealers(df_future):
    df_trader_vol= pd.DataFrame(df_future.groupby('dealer')['volume'].sum())
    df_trader_vol = df_trader_vol.sort_values(by='volume', ascending=False)
    df =pd.DataFrame(columns=['n_top_dealers', 'percent_trade_volume'])
    total_trade_vol=df_future['volume'].sum()
    for n in range(0, len(df_future['dealer'].unique()),5):
        df.loc[len(df)] ={"n_top_dealers":n,"percent_trade_volume":df_trader_vol['volume'][:n].sum()/total_trade_vol}
    
    return df

def cal_IC(df_idx, df_future):
    future_cols=df_future.columns.drop(['dealer'])
    future_cols_without_dealer=df_future.columns.drop(['dealer'])
    df_future_top_by_date=df_future[future_cols_without_dealer].groupby(df_future.index).sum()
    ics = df_future_top_by_date[future_cols][:-1].corrwith(df_idx['Return'][:-1], method='pearson')
    df_ic=pd.DataFrame(ics)
    df_ic.reset_index(inplace=True)
    df_ic.columns=['Column','IC']
    return df_ic

def cal_corr(df_idx,df_future_top):
    df_corr= pd.DataFrame(columns=['dealer','column', 'Correlation', 'Sum'])
    top_dealers=df_future_top['dealer'].unique()
    future_cols=df_future_top.columns.drop(['dealer'])
    for dealer in top_dealers:
        future_cols=list(df_future_top.columns.drop(['dealer']).values)
        df_future_dealer=df_future_top[df_future_top['dealer']==dealer]
        df_idx_future_dealer =df_idx.merge(df_future_dealer, left_index=True, right_index=True, how='left')
        df_idx_future_dealer=df_idx_future_dealer[future_cols +['Return']]

        df_idx_future_dealer=df_idx_future_dealer[:-1].fillna(0)
        df_corr_dealer=df_idx_future_dealer[future_cols].corrwith(df_idx_future_dealer['Return'])
                
        for x in df_corr_dealer.index:
            df_corr.loc[len(df_corr)] ={"dealer":dealer,'column':x, 'Correlation':df_corr_dealer[x], 'Sum':df_future_dealer[x].sum()}
        
    return df_corr


if __name__ == "__main__":
    df = read_yahoo_data('000300.SS', '2020-01-01', '2025-12-31')
    df = add_technical_indicators(df)
    df = add_technical_return(df)
    print(df.head())
    print("done")
    