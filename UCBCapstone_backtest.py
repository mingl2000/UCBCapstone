from app_settings import *
set_random_seed()

import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime
import yfinance as yf
from UCBCapstone_data_io import  *
from UCBCapstone_data_prepare import  *
from UCBCapstone_data_view import *
from UCBCapstone_models import *

import math
class BuyAndHoldStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        pass
        #dt = dt or self.datas[0].datetime.datetime(0)
        #print(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {txt}")
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
    def next(self):
        if self.order:
            return
        if not self.position:
            self.order=self.buy()
    def stop(self):
        self.order=self.close()
        #portfolio_value = self.broker.getvalue()

class BackTestStrategy_one_step(bt.Strategy):
    params = (
        ('model_name',''),
        ('model', None),
        ('product',None),
        ('model_data', None),
        ('verbose',None),
    )
    def __init__(self):
        self.dataclose = self.datas[0].close        
        self.order = None
        self.counter=0
        self.pnl=0
    def next(self):
        if len(self.datas[0]) == self.datas[0].buflen() - 1:  # Last original bar
            for data in self.datas:
                if self.getposition(data).size != 0:
                    self.close(data=data)
                    self.log(f'Closing position for {data._name} at {self.dataclose[0]:.2f}')
            return
        
        if self.broker.getvalue()<0:
            return

        model_name=self.params.model_name
        model =self.params.model
        product =self.params.product
        model_data =self.params.model_data
        verbose =self.params.verbose


        current_data =model_data[self.counter:self.counter+1]

        if "tensorflow" in model_name.lower(): 
            predicted_ret=model.predict(current_data, verbose=0).flatten()[0]
        else:
            predicted_ret=model.predict(current_data).flatten()[0]

        if not self.position:
            if predicted_ret>0 or predicted_ret==True:
                self.order = self.buy()
        else:
            if predicted_ret < 0:
                self.order = self.sell()
        
        self.counter=self.counter+1

    def notify_order(self, order):
        if order.status == order.Completed:
            
            if order.OrdTypes[order.ordtype]=='Sell':
                self.pnl = self.pnl +order.executed.pnl
                self.log(f"Trade closed: PnL (gross):   {order.executed.pnl:.2f},  Cash:   {self.broker.get_cash():2f}      All trades pnl:    {self.pnl:.2f}")
            else:
                self.log(f"Buy executed: {order.OrdTypes[order.ordtype]} product:   {self.params.product}   Price:   {order.executed.price:.3f}    Size:  {order.executed.size}  Cash:  {self.broker.get_cash():.2f}")
                #self.position_size=order.executed.size

        elif order.status in [order.Canceled, order.Rejected, order.Margin]:
            self.log(f"Order failed: {order.status}, Price={order.executed.price}")

    def log(self, txt, dt=None):
        if self.params.verbose>0 :
            dt = dt or self.datas[0].datetime.date(0)
            self.log(f'{dt.isoformat()}, {txt}')


class BackTestStrategy_n_step(bt.Strategy):
    params = (
        ('model_name',''),
        ('model', None),
        ('product',None),
        ('model_data', None),
        ('sequence_length', 5), 
        ('verbose',None),

    )
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.counter=0
        self.pnl=0
    def next(self):
        if len(self.datas[0]) == self.datas[0].buflen() - 1:  # Last original bar
            for data in self.datas:
                if self.getposition(data).size != 0:
                    self.close(data=data)
                    self.log(f'Closing position for {data._name} at {self.dataclose[0]:.2f}')
            return

        model_name=self.params.model_name
        model =self.params.model
        product =self.params.product
        model_data =self.params.model_data
        verbose =self.params.verbose
        sequence_length = self.params.sequence_length
        
        current_data =model_data[(self.counter+1-sequence_length):(self.counter+1)]
        ds_pred = tf.keras.preprocessing.timeseries_dataset_from_array(current_data, None, sequence_length=sequence_length)
        predicted_ret= model.predict(ds_pred, verbose=0).flatten()[0]


        if not self.position:
            if predicted_ret>0 or predicted_ret==True:
                self.order = self.buy()
        else:
            if predicted_ret < 0:
                self.order = self.sell()
        
        self.counter=self.counter+1


    def notify_order(self, order):

        if order.status == order.Completed:
            
            if order.OrdTypes[order.ordtype]=='Sell':
                self.pnl = self.pnl +order.executed.pnl
                self.log(f"Trade closed: PnL (gross):   {order.executed.pnl:.2f},  Cash:   {self.broker.get_cash():2f}      All trades pnl:    {self.pnl:.2f}")
            else:
                self.log(f"Buy executed: {order.OrdTypes[order.ordtype]} product:   {self.params.product}   Price:   {order.executed.price:.3f}    Size:  {order.executed.size}  Cash:  {self.broker.get_cash():.2f}")

    def log(self, txt, dt=None):
        if self.params.verbose>0 :
            dt = dt or self.datas[0].datetime.date(0)
            self.log(f'{dt.isoformat()}, {txt}')


def backtest(product, model_name, model,X_train, y_train, X_val, y_val, X_test, y_test,df_close_test, classification, start_date, end_date,verbose=1, epochs=200):

    sequence_length=5
    cerebro = bt.Cerebro()
    if model_name=='BuyAndHold':
        cerebro.addstrategy(BuyAndHoldStrategy)
        ic_from_model=0
        mse=0
        test_score=0
    elif model_name =='Tensorflow_MLP':
        model_name, model, ic_from_model,mse,history,test_score= train_and_IC_TensorFlow_MLP(X_train, y_train, X_val, y_val, X_test, y_test, classification,timesteps=1, epochs=epochs)
        cerebro.addstrategy(BackTestStrategy_one_step, model_name=model_name, model=model, product=product, model_data=X_test, verbose=verbose)
    elif model_name =='Tensorflow_RNN':
        model_name, model, ic_from_model,mse,history, test_score = train_and_IC_TensorFlow_RNN(X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps=sequence_length, epochs=epochs)
        cerebro.addstrategy(BackTestStrategy_n_step, model_name=model_name, model=model, product=product,model_data=X_test, sequence_length=sequence_length, verbose=verbose)
    elif model_name =='Tensorflow_LTSM':
        model_name, model, ic_from_model,mse,history, test_score = train_and_IC_TensorFlow_LTSM(X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps=sequence_length, epochs=epochs)
        cerebro.addstrategy(BackTestStrategy_n_step, model_name=model_name, model=model, product=product, model_data=X_test, sequence_length=sequence_length, verbose=verbose)

    else: # Non-tensorflow models
        model_name, model, ic_from_model,mse, test_score =train_and_IC(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, classification)
        cerebro.addstrategy(BackTestStrategy_one_step, model_name=model_name, model=model, product=product, model_data=X_test, verbose=verbose)

    
    df_close_test.columns=['close','high','low','open','volume']
    data = bt.feeds.PandasData(dataname=df_close_test)
    cerebro.adddata(data)
    
    # Set initial cash and commission
    initial_captal=100000.0
    cerebro.broker.setcash(initial_captal)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=99)
    
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    # Run strategy
    results = cerebro.run()
    start_date=min(df_close_test.index)
    end_date=max(df_close_test.index)
    account_val =cerebro.broker.getvalue() if cerebro.broker.getvalue()>0 else 0 
    
    annual_return= annualized_return(account_val/initial_captal-1,start_date, end_date)
    sharperatio = results[0].analyzers.sharpe.get_analysis()['sharperatio']
    if sharperatio is None:
        sharperatio=0
    max_drawdown = results[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    
    

    trade_analyzer = results[0].analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analyzer.get('total', {}).get('closed', 0)
    winners = trade_analyzer.get('won', {}).get('total', 0)
    win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
    print(f'{model_name:>30}: product:   {product}   Final Portfolio Value:    {account_val:.2f}     annual_return: {annual_return*100:.2f}%     win_rate: {win_rate:.2f}%   total_trades: {total_trades:>3}  Sharpe Ratio: {sharperatio:.2f}      max drawdown: {max_drawdown:.2f}')
    # Plot results
    if verbose>0:
        figs = cerebro.plot(iplot=False)  # Force non-interactive
        plt.show()  
    
    if ic_from_model is None:
        ic_from_model=0

    return model_name, ic_from_model,mse, test_score, annual_return, sharperatio, max_drawdown, win_rate, total_trades
    pass


def annualized_return(total_return, start_date, end_date):
    # Calculate number of years
    days = (end_date - start_date).days
    years = days / 365.25  # Use 365.25 to account for leap years
    
    # Calculate annualized return using CAGR formula
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized


def backtest_all_products(products='IH', important_features=[], classification=False, verbose=1, epochs=400):
    df_result =pd.DataFrame(columns=["product",'model_name','IC%','MSE','test_score', 'annual_return%','win_rate','total_trades', 'sharperatio','max_drawdown%', 'classification'])
    if classification:
        models = classification_models
    else:
        models = regression_models
    for product in products.split(','):
        X_train, y_train, X_val, y_val, X_test, y_test,df_close_test, start_date, end_date = load_and_preprocess_and_split_data(product,important_features, classification)

        # Buy and hold strategy
        model_name='BuyAndHold'
        model_name, ic,mse,test_score, annual_return, sharperatio, max_drawdown,win_rate, total_trades =backtest(product, model_name, None,X_train, y_train, X_val, y_val, X_test, y_test, df_close_test,classification, start_date, end_date, verbose=0)   
        df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, 'IC%':None,'MSE':None, 'test_score':None, "annual_return%":annual_return*100,'win_rate':win_rate,'total_trades':total_trades, 'sharperatio':sharperatio,'max_drawdown%':max_drawdown,'classification':classification}
        # None tensorflow models
        for model_name, model in models.items():
            model_name, ic,mse,test_score, annual_return, sharperatio, max_drawdown,win_rate, total_trades =backtest(product, model_name, model,X_train, y_train, X_val, y_val, X_test, y_test, df_close_test,classification, start_date, end_date, verbose=verbose)    
            if classification:
                df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, 'IC%':None,'MSE':None, 'test_score':test_score, "annual_return%":annual_return*100,'win_rate':win_rate,'total_trades':total_trades, 'sharperatio':sharperatio,'max_drawdown%':max_drawdown,'classification':classification}
            else:        
                df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, 'IC%':ic*100,'MSE':mse, 'test_score':None, "annual_return%":annual_return*100,'win_rate':win_rate,'total_trades':total_trades, 'sharperatio':sharperatio,'max_drawdown%':max_drawdown,'classification':classification}
        # Tensorflow models
        '''
        for model_name in ['Tensorflow_MLP']:
            model_name, ic,mse,test_score, annual_return, sharperatio, max_drawdown,win_rate, total_trades =backtest(product, model_name, None, X_train, y_train, X_val, y_val, X_test, y_test, df_close_test, classification, start_date, end_date, verbose=verbose,epochs=epochs)
            if classification:
                df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, 'IC%':None,'MSE':None, 'test_score':test_score, "annual_return%":annual_return*100,'win_rate':win_rate,'total_trades':total_trades, 'sharperatio':sharperatio,'max_drawdown%':max_drawdown,'classification':classification}
            else:
                df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, 'IC%':ic*100,'MSE':mse, 'test_score':None, "annual_return%":annual_return*100,'win_rate':win_rate,'total_trades':total_trades, 'sharperatio':sharperatio,'max_drawdown%':max_drawdown,'classification':classification}
        '''
    return df_result
# Main execution
if __name__ == '__main__':
    products='IC'
    #df_result= train_predict_one_model(products, 'VotingRegressor')
    #voting_model=get_model(df_result, 'VotingRegressor').values[0]
    #feature_names=get_deature_names(df_result)
    #important_features= get_top_feature_importance(voting_model, feature_names, 50)
    important_features=[]
    classification=False
    product='IC'
    important_features=[]
    classification=False

    #X_train, y_train, X_val, y_val, X_test, y_test,df_close_test, start_date, end_date = load_and_preprocess_and_split_data(product, important_features, classification)
    #model_name='BuyAndHold'
    #model_name, ic,mse,test_score, annual_return, sharperatio, max_drawdown,win_rate, total_trades =backtest(product, model_name, None,X_train, y_train, X_val, y_val, X_test, y_test, df_close_test,classification, start_date, end_date, verbose=0)   

    df_backtest_result= backtest_all_products(products,important_features, classification, verbose=0, epochs=75)
    print(df_backtest_result)



    