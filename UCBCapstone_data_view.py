import mplfinance as mpf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def plot_csi_index(df):
    """
    Plot the historical data of the index future.
    
    Parameters:
    df_future_index_history (DataFrame): DataFrame containing the historical data of the index future.
    """
    if df.empty:
        print("No data to plot.")
        return
    
    # Ensure the DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('Date', inplace=True)
    
    # Plotting
    apdict = [
        mpf.make_addplot(df[-500:]['MACD'], panel=2, type='bar', ylabel='MACD', color='dodgerblue'),        
    ]

    mpf.plot(df[-500:], addplot=apdict,type='candle', style='charles', title='CSI index', ylabel='Price', volume=True, mav=(20, 50), show_nontrading=True, figsize=(20, 6),tight_layout=True)





def plot_future_trading(df_idx, df_trading, df_future_by_date):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 12), sharex=True)

# Step 3: Plot df on the first subplot
    sns.lineplot(data=df_idx[['Return']], ax=ax1)
    ax1.set_title('CSI 300 index change percent next day')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Values')
    ax1.tick_params(axis='x', rotation=45)

    # Step 4: Plot df3 on the second subplot

    sns.lineplot(data=df_trading, ax=ax2)
    ax2.set_title('CSI 300 Index Future Contract Trading Top Volumes Info')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Values')
    ax2.tick_params(axis='x', rotation=45)
    sns.lineplot(data=df_future_by_date, ax=ax2)
    ax2.legend(labels=df_future_by_date.columns, title='Columns', loc='upper left')
    ax2.tick_params(axis='x', rotation=45)

    # Step 5: Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_one(df, char_type,x_col, y_col, title, color='skyblue'):
    plt.figure(figsize=(20, 5))
    if char_type=='barplot':
        sns.barplot(x=x_col, y=y_col, data=df, color=color)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

def plot_n(df, n, x_col, y_cols, titles,colors):
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(20, 12))
    #for ax in list(axes):
    for i in range(n):
        sns.barplot(data=df, x=x_col, y=y_cols[i],ax=axes[i], color=colors[i])
        axes[i].set_title(titles[i])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


def plot_tensorflow_train_history(df_result, model_name, classification=False):
    for index, row in df_result[df_result['model_name']==model_name].iterrows():
        model_name = row['model_name']      # Access the model object
        history = row['history']
        if history is None:
            continue
        # Create a chart
        plt.figure(figsize=(10, 5))

        # Plot RMSE
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label=f'{model_name} Train loss')
        plt.plot(history.history['val_loss'], label=f'{model_name} Validation loss')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()

        # Plot MAE
        plt.subplot(1, 2, 2)
        if classification:
            plt.plot(history.history['accuracy'], label=f'{model_name} Train accuracy')
            plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation accuracy')
            plt.yscale('log')
            plt.ylabel('Accuracy')
        
        else:
            plt.plot(history.history['mae'], label=f'{model_name} Train MAE')
            plt.plot(history.history['val_mae'], label=f'{model_name} Validation MAE')
            plt.yscale('log')
            plt.ylabel('MAE')
        plt.xlabel('Iteration')
        plt.legend()

    plt.tight_layout()
    plt.show()

def print_models(df_result):
    for index, row in df_result.iterrows():
        model_name = row['model_name']      # Access the model object
        model = row['model']  
        if model_name.startswith('Tensorflow'):
            print(f'----------- {model_name} ----------')
            model.summary()
            print(f'----------- end of {model_name} ----------')
def get_model(df_result, model_name):
    return df_result[df_result['model_name']==model_name]['model']

def plot_feature_importance(voting_model,feature_names):
    importance_dfs = {}

    for name, model in voting_model.named_steps['voting_clf'].named_estimators_.items():
        print(f"\nFeature Importance for {name}:")
        importance_df=None
        if name in ['LinearRegression', 'RidgeRegression']:
            coef= model.steps[-1][-1].coef_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coef
            }).sort_values(by='Importance', ascending=False)
        
        elif name in ['DecisionTreeRegressor', 'AdaBoostRegressor', 'XGBRegressor']:
            coef= model.steps[-1][-1].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coef
            }).sort_values(by='Importance', ascending=False)
        
        elif name == 'TransformedTargetRegressor':
            coef = model.regressor_.coef_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coef
            }).sort_values(by='Importance', ascending=False)
        else:
            continue
        if importance_df is not None:
            importance_dfs[name] = importance_df
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Feature', y='Importance', data=importance_df[:25])
            plt.title(f'Feature Importance for {name}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
    plt.show()    

def get_deature_names(df_result):
    return df_result[df_result['model_name']=='VotingRegressor']['feature_names'].to_numpy()[0]