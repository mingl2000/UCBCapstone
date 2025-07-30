from app_settings import set_random_seed,set_global_settings
set_random_seed()
set_global_settings()

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, VotingRegressor, VotingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from UCBCapstone_data_view import *
from UCBCapstone_data_io import  *
from UCBCapstone_data_prepare import  *

from UCBCapstone_models import  *


def grid_search_voting_regressor(X_train, y_train):
    voter = VotingRegressor(
        estimators=[(name, model) for name, model in regression_models.items()]
    )
    param_grid = {
    'LinearRegression__lr__fit_intercept': [True, False],
    'RidgeRegression__ridge__alpha': [0.1, 1.0, 10.0],  #Best parameters: {'RidgeRegression__ridge__alpha': 10.0}
    
    'KnnRegressor__knn__n_neighbors': [5, 7, 9], #Best parameters: {'KnnRegressor__knn__n_neighbors': 9}
    'KnnRegressor__knn__weights': ['uniform', 'distance'], #Best parameters: {'KnnRegressor__knn__n_neighbors': 9, 'KnnRegressor__knn__weights': 'distance'}
    
    'DecisionTreeRegressor__dt__max_depth': [3, 5, None],
    'DecisionTreeRegressor__dt__min_samples_split': [2, 5],#Best parameters: {'DecisionTreeRegressor__dt__max_depth': 5, 'DecisionTreeRegressor__dt__min_samples_split': 2}
    
    'AdaBoostRegressor__ada__n_estimators': [50, 100],
    'AdaBoostRegressor__ada__learning_rate': [0.01, 0.1], #Best parameters: {'AdaBoostRegressor__ada__learning_rate': 0.1, 'AdaBoostRegressor__ada__n_estimators': 50}
    
    'XGBRegressor__xgb__max_depth': [3, 5],
    'XGBRegressor__xgb__learning_rate': [0.01, 0.1],
    'XGBRegressor__xgb__n_estimators': [100, 200], #Best parameters: {'XGBRegressor__xgb__learning_rate': 0.01, 'XGBRegressor__xgb__max_depth': 3, 'XGBRegressor__xgb__n_estimators': 100}
    'weights': [
        [1, 1, 1, 1, 1, 1, 1,1],  # Equal weights
        [2, 1, 1, 1, 1, 1, 1,1],  # Emphasize LinearRegression
        [1, 1, 2, 1, 1, 1, 1,1]   # Emphasize KnnRegressor
    ]}

    grid_search = GridSearchCV(
            estimator=voter,
            param_grid=param_grid,
            cv=2,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
    grid_search.fit(X_train, y_train)
    results = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'best_estimator': grid_search.best_estimator_
    }
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    return grid_search.best_params_

def grid_search_voting_classifier(X_train, y_train):
    voter = VotingClassifier(
        estimators=[(name, model) for name, model in classification_models.items()]
    )

    param_grid = {
        'LogisticRegression__lg__fit_intercept': [True,False],
        'KNeighborsClassifier__knn__n_neighbors': [3,5,7],
        'KNeighborsClassifier__knn__weights': ['uniform', 'distance'], # {'KNeighborsClassifier__knn__n_neighbors': 7, 'KNeighborsClassifier__knn__weights': 'uniform', 'LogisticRegression__lg__fit_intercept': True}
        'AdaBoostClassifier__ada__n_estimators': [100,200],
        'AdaBoostClassifier__ada__learning_rate': [0.1,0.01], #Best parameters: {'AdaBoostClassifier__ada__learning_rate': 0.1, 'AdaBoostClassifier__ada__n_estimators': 100, 'KNeighborsClassifier__knn__n_neighbors': 7, 'KNeighborsClassifier__knn__weights': 'uniform', 'LogisticRegression__lg__fit_intercept': True}

        'XGBClassifier__xgb__max_depth': [3,5,7],
        'XGBClassifier__xgb__learning_rate': [0.1,0.01],
        'XGBClassifier__xgb__n_estimators': [100,200], #Best parameters: {'XGBRegressor__xgb__learning_rate': 0.01, 'XGBRegressor__xgb__max_depth': 3, 'XGBRegressor__xgb__n_estimators': 100}
        'weights': [[1, 1, 1, 1,1], [2, 1, 1, 1,1], [1, 2, 1, 1,1], [1, 1, 2, 1,1 ], [1, 1, 1, 2,1],[1, 1, 1, 1,2]] #Best parameters: {'AdaBoostClassifier__ada__learning_rate': 0.1, 'AdaBoostClassifier__ada__n_estimators': 100, 'KNeighborsClassifier__knn__n_neighbors': 7, 'KNeighborsClassifier__knn__weights': 'uniform', 'LogisticRegression__lg__fit_intercept': True, 'XGBClassifier__xgb__learning_rate': 0.1, 'XGBClassifier__xgb__max_depth': 3, 'XGBClassifier__xgb__n_estimators': 100, 'weights': [1, 1, 2, 1, 1]}
    }
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    grid_search = GridSearchCV(
            estimator=voter,
            param_grid=param_grid,
            cv=2,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
    grid_search.fit(X_train, y_train)
    results = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'best_estimator': grid_search.best_estimator_
    }
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    return grid_search.best_params_


def main():
    classification=False
    product='IF'
    X_train, y_train, X_val, y_val, X_test, y_test,df_close, start_date,last_date =load_and_preprocess_and_split_data(product,[], classification)
    if classification:
        grid_search_voting_classifier(X_train, y_train)
    else:
        best_params = grid_search_voting_regressor(X_train, y_train)
    

if __name__ == '__main__':
    main()