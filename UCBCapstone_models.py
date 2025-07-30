from app_settings import set_random_seed,set_global_settings
set_random_seed()
set_global_settings()

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, VotingRegressor, VotingClassifier
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
import shutil
import keras_tuner as kt
import math

from scipy.stats import spearmanr
from UCBCapstone_data_view import *
from UCBCapstone_data_io import  *
from UCBCapstone_data_prepare import  *
'''
regression_models
Best parameters: 
        {'AdaBoostRegressor__ada__learning_rate': 0.1, 
        'AdaBoostRegressor__ada__n_estimators': 50, 
        'DecisionTreeRegressor__dt__max_depth': 5, 
        'DecisionTreeRegressor__dt__min_samples_split': 2, 
        'KnnRegressor__knn__n_neighbors': 9, 
        'KnnRegressor__knn__weights': 'distance', 
        'LinearRegression__lr__fit_intercept': False, 
        'RidgeRegression__ridge__alpha': 10.0, 
        'XGBRegressor__xgb__learning_rate': 0.01, 
        'XGBRegressor__xgb__max_depth': 3, 
        'XGBRegressor__xgb__n_estimators': 100, 
        'weights': [1, 1, 2, 1, 1, 1, 1, 1]}
'''
regression_models = {
    'LinearRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression(fit_intercept=False))]),

    'RidgeRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10, random_state=42))]),
    'KnnRegressor': Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=9,weights='distance' ))]),
    'DecisionTreeRegressor': Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeRegressor(max_depth=5, min_samples_split=2, random_state=42))]),
    'TransformedTargetRegressor': TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler()
    ),
    'AdaBoostRegressor':Pipeline([
        ('scaler', StandardScaler()),
        ('ada', AdaBoostRegressor(learning_rate=0.1, n_estimators=50,random_state=42))
    ]),
    'XGBRegressor': Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.01, n_estimators=100, eval_metric='rmse'))
    ])
}


regression_models['VotingRegressor'] = Pipeline([
    ('scaler', StandardScaler()),  # Scale features for all classifiers
    ('voting_clf', VotingRegressor(estimators=[(name, model) for name, model in regression_models.items()]))
])


'''
# based on gridsearchCV
Best parameters: {
        'AdaBoostClassifier__ada__learning_rate': 0.1, 
        'AdaBoostClassifier__ada__n_estimators': 100, 
        'KNeighborsClassifier__knn__n_neighbors': 7, 
        'KNeighborsClassifier__knn__weights': 'uniform', 
        'LogisticRegression__lg__fit_intercept': True, 
        'XGBClassifier__xgb__learning_rate': 0.1, 
        'XGBClassifier__xgb__max_depth': 3, 
        'XGBClassifier__xgb__n_estimators': 100, 
        'weights': [1, 1, 2, 1, 1]}

'''
classification_models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('lg', LogisticRegression( fit_intercept=True, max_iter=1000))]),

    'KNeighborsClassifier': Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=7, weights='uniform'))]),
    'AdaBoostClassifier':Pipeline([
        ('scaler', StandardScaler()),
        ('ada', AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42))
    ]),
    'XGBClassifier': Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(learning_rate=0.1, random_state=42, max_depth=3, n_estimators=100, eval_metric='logloss'))
    ])
}


classification_models['VotingClassifier'] = Pipeline([
    ('scaler', StandardScaler()),  # Scale features for all classifiers
    ('voting_clf', VotingClassifier(
    estimators=[(name, model) for name, model in classification_models.items()],
    voting='soft'  # Use 'hard' for majority voting, 'soft' for probability averaging
))])


def cal_IC_from_prediction(model,X_test, y_test, classification=False):
    test_score=None
    ic=None
    mse=None
    y_test_pred = model.predict(X_test)
    if classification:
        test_score= accuracy_score(y_test_pred, y_test)
    else:
        y_test_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        if np.std(y_test_pred) > 0:
            ic, _=spearmanr(y_test_pred, y_test)
        else:
            ic=0
    return ic, mse, test_score 

def do_prediction_tf(model, X_pred, y_normalizer): 
    y_pred = model.predict(X_pred)
    y_pred_scaled = model.predict(X_pred, verbose=0).flatten()
    y_pred = y_normalizer(y_pred_scaled)
    return y_pred[0]

def load_and_preprocess_data(product):
    np.random.seed(42)
    df_index_future, df_future_trading, df_index =read_all(product)
    df_index= add_technical_indicators(df_index)
    df_index= add_technical_return(df_index)

    start_date=min(df_index_future.index)
    last_date=max(df_index_future.index)

    df_future_top=get_idx_future_top_traders(df_future_trading,50)
    df_future_pivoted =pivote_df_future(df_future_top)
    df_idx_with_future_trades =df_index.merge(df_future_pivoted, left_index=True, right_index=True, how='left').fillna(0)
    df_idx_with_future_trades.dropna(inplace=True)
    df_close =df_idx_with_future_trades[['Close','High','Low','Open','Volume']]
    #X= df_idx_with_future_trades.drop(columns=['Close','Return','Open','High','Low','Volume'])
    X= df_idx_with_future_trades.drop(columns=['Return'])
    y=df_idx_with_future_trades['Return']
    return X,y, df_close, start_date, last_date  


from scipy.stats import pearsonr


class EarlyStoppingByValLoss(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(EarlyStoppingByValLoss, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_mae')
        if val_loss is not None and val_loss < self.threshold:
            print(f'\nStopping search: val_loss ({val_loss:.4f}) is below threshold ({self.threshold:.4f})')
            self.model.stop_training = True


def train_and_IC_TensorFlow_MLP(X_train, y_train, X_val, y_val, X_test, y_test,  classification=False,  timesteps=1, epochs=200):

    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(X_train.values.astype(np.float32))
    # MLP Model
    model = tf.keras.Sequential([
        normalization_layer,
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2,seed=42),
       tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2,seed=42),
        tf.keras.layers.Dense(32, activation='relu'),
    ])
    learning_rate=0.0001
    if classification:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        loss='binary_crossentropy'
        metrics=['accuracy']
    else:
        model.add(tf.keras.layers.Dense(1))
        loss='mean_squared_error'
        #loss='mae'
        metrics=['mae']
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    #optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)

    reduced_lr_callback=tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Metric to monitor
        factor=0.1,          # Factor by which the learning rate will be reduced
        patience=5,          # Number of epochs with no improvement after which LR will be reduced
        verbose=1,           # 0: quiet, 1: update messages
        min_lr=0.00001       # Minimum learning rate
    )

    history= model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0, callbacks=[EarlyStoppingByValLoss(threshold=0.03),reduced_lr_callback])
    
    if classification:
        test_loss, test_score = model.evaluate(X_test, y_test, verbose=0)
        ic=None
        mse=None
    else:
        ic,mse, test_score= cal_IC_from_prediction(model,X_test, y_test, classification)
    

    return 'Tensorflow_MLP', model, ic, mse, history, test_score

def extract_arrays_from_timeseries_dataset(ds):
    y_extracted = []
    X_extracted = []
    for batch in ds:
        inputs, targets = batch  # Each batch contains (inputs, targets)
        y_extracted.append(targets.numpy())  # Convert targets tensor to NumPy array
        X_extracted.append(inputs.numpy())

    y_extracted = np.concatenate(y_extracted, axis=0)
    X_extracted = np.concatenate(X_extracted, axis=0)

    return X_extracted, y_extracted

def train_and_IC_TensorFlow_RNN_LTSM(model_name, model, X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps, epochs=200, learning_rate=0.0001):

    ds_train = tf.keras.preprocessing.timeseries_dataset_from_array(X_train, y_train, sequence_length=timesteps)
    ds_val = tf.keras.preprocessing.timeseries_dataset_from_array(X_val, y_val, sequence_length=timesteps)
    ds_test = tf.keras.preprocessing.timeseries_dataset_from_array(X_test, y_test, sequence_length=timesteps)

    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(X_train.values.astype(np.float32))
    model = tf.keras.Sequential([normalization_layer] + [layer for layer in model.layers])
    # Compile models

    if classification:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        loss='binary_crossentropy'
        metrics=['accuracy']
    else:
        model.add(tf.keras.layers.Dense(1))
        loss='mean_squared_error'
        metrics=['mae']

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)



    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose=1)

    # Evaluate models
    rnn_loss, rnn_mae = model.evaluate(ds_test, verbose=0)

    X_extracted, y_extracted = extract_arrays_from_timeseries_dataset(ds_test)
    ic,mse, test_score= cal_IC_from_prediction(model,X_extracted, y_extracted)

    return model_name, model, ic, mse, history, test_score

def train_and_IC_TensorFlow_RNN(X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps, epochs=200):
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(96, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.SimpleRNN(32, activation='relu'),
    ])
    if classification:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(1))


    return train_and_IC_TensorFlow_RNN_LTSM('Tensorflow_RNN', model, X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps, epochs)

def train_and_IC_TensorFlow_LTSM(X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps, epochs=200):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(1024, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(256, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='relu'),
    ])
    if classification:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    else:
        model.add(tf.keras.layers.Dense(1))

    return train_and_IC_TensorFlow_RNN_LTSM('Tensorflow_LTSM', model, X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps,epochs)

def train_and_IC(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, classification=False):
    model.fit(X_train, y_train)
    ic,mse, test_score= cal_IC_from_prediction(model,X_test, y_test, classification)
    return model_name, model, ic, mse, test_score

def load_and_preprocess_and_split_data(product, features=[], classification=False):
    X,y, df_close, start_date,last_date = load_and_preprocess_data(product)
    if classification:
        y=y>0
    X.columns=[str(col).lower().strip() for col in X.columns]
    if len(features)>0:
        X=X[features]
    n = len(X)
    X_train = X[0:int(n*0.85)]
    y_train = y[0:int(n*0.85)]
    X_val = X[int(n*0.85):int(n*0.95)]
    y_val = y[int(n*0.85):int(n*0.95)]
    X_test = X[int(n*0.95):]
    y_test = y[int(n*0.95):]
    df_close_test=df_close[int(n*0.95):]
    return X_train, y_train, X_val, y_val, X_test, y_test,df_close_test, start_date,last_date


def train_predict_all_models(product, important_features=[], classification=False, epochs=200):
    
    df_result =pd.DataFrame(columns=["product",'model_name','IC%','rmse','test_score','prediction%','start_date','last_date','model','history','feature_names'])
    print(f'Train, test and predict on CSI future contract:"{product} classification={classification}"...')
    X_train, y_train, X_val, y_val, X_test, y_test,df_close, start_date,last_date =load_and_preprocess_and_split_data(product,important_features,classification)

    '''
        Non-Tensorflow models
    '''
    if classification:
        models= classification_models
    else:
        models= regression_models
    for model_name, model in models.items():
        model_name, model, ic_from_model,mse, test_score =train_and_IC(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,classification)
        if classification:
            df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, "model":model,'IC%':None,'rmse':None,'test_score':test_score, 'start_date':start_date,'last_date':last_date, "history":None, 'feature_names':X_train.columns}
        else:
            df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, "model":model,'IC%':ic_from_model*100,'rmse':math.sqrt(mse),'test_score':None,'start_date':start_date,'last_date':last_date, "history":None, 'feature_names':X_train.columns}
    ''' 
        Tensorflow models 
    '''    
    for train_and_IC_func in [ train_and_IC_TensorFlow_MLP]:
        model_name, model, ic_from_model,mse,history,test_score = train_and_IC_func(X_train, y_train, X_val, y_val, X_test, y_test, classification, timesteps=5, epochs=epochs)
        if classification:
            df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, "model":model,'IC%':None,'rmse':None,'test_score':test_score, 'start_date':start_date,'last_date':last_date, "history":history, 'feature_names':X_train.columns}
        else:
            df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, "model":model,'IC%':ic_from_model*100,'rmse':math.sqrt(mse),'test_score':None,'start_date':start_date,'last_date':last_date, "history":history, 'feature_names':X_train.columns}
    return df_result

def train_predict_one_model(product, model_name, classification):
    df_result =pd.DataFrame(columns=["product",'model_name','IC%','rmse','test_score','prediction%','start_date','last_date','model','history','feature_names'])
    print(f'Train, test and predict on CSI future contract:"{product}"...')
    X_train, y_train, X_val, y_val, X_test, y_test,df_close, start_date,last_date =load_and_preprocess_and_split_data(product,[], classification)
    '''
        Non-Tensorflow models
    '''
    if classification:
        model=classification_models[model_name]
    else:
         model=regression_models[model_name]
    model_name, model, ic_from_model,mse, test_score =train_and_IC(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test,classification)
    if classification:
        df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, "model":model,'IC%':None,'rmse':None,'test_score':test_score, 'start_date':start_date,'last_date':last_date, "history":None, 'feature_names':X_train.columns}
    else:
        df_result.loc[len(df_result)] = {"product": product,"model_name":model_name, "model":model,'IC%':ic_from_model*100,'rmse':math.sqrt(mse),'test_score':None, 'start_date':start_date,'last_date':last_date, "history":None, 'feature_names':X_train.columns}

    return df_result


def main(product='IH'):
    ''' regression'''
    df_result= train_predict_one_model(product, 'VotingRegressor',classification=False)
    voting_model=get_model(df_result, 'VotingRegressor').values[0]
    feature_names=get_deature_names(df_result)
    #important_features= get_top_feature_importance(voting_model, feature_names, 100)
    #important_features=[]  # empty means all

    df_result = train_predict_all_models(product, feature_names)
    print(df_result)
    plot_tensorflow_train_history(df_result,'Tensorflow_MLP')
    print_models(df_result[['model_name','model']])
    
    plot_feature_importance(voting_model, feature_names)
    ''' classifier'''
    model_name ='VotingClassifier'
    df_result= train_predict_one_model(product,model_name, classification=True)
    voting_model=get_model(df_result, model_name).values[0]
    important_features=[]  # empty means all

if __name__ == "__main__":
    product='IF'
    #train_predict_all_models(product, important_features=[], classification=False)
    classification=False
    df_train_test_result= train_predict_all_models(product, important_features=[], classification=classification, epochs=2000)
    for model_name in ['Tensorflow_MLP']:
        #model=get_model(df_train_test_result, model_name)
        #model.values[0].summary()
        plot_tensorflow_train_history(df_train_test_result, model_name,classification=classification)


    #main()