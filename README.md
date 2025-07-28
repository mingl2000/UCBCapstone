# Trading Index ETFs Using Predictions from Related Index Futures Contract Trading Information

**Author: Ming Lu**

## Executive Summary
This report investigates the development and evaluation of predictive models for trading market index Exchange-Traded Funds (ETFs) using historical data from related index futures contracts and market index performance. By leveraging trading information from the top 20 futures traders across seven categories, multiple machine learning models were trained to identify profitable trading opportunities. Backtesting results demonstrate significant performance improvements over traditional buy-and-hold strategies, particularly for certain ETFs. The report details the methodology, key findings, and recommendations for practical implementation.

## Key Findings
- **High Predictive Accuracy**: A strong Information Coefficient (IC%) was observed between predicted and actual ETF returns in the unseen test dataset, validating the models' predictive power.
- **Significant Performance Improvement**:
  - Trading the ETF 510500.SS (CSI 500 Index) using futures data from the IC contract yielded an annual return of **33.55%** with the VotingRegressor model, compared to **0.5%** for a buy-and-hold strategy.
  - A portfolio of four index ETFs (equally weighted) using the VotingRegressor model achieved an annual return of **23.25%**, surpassing the buy-and-hold return of **15.15%**.
- **Classification Model Insights**: Classification models triggered only one trade signal during a two-month test period, raising questions about whether this is an incidental outcome or a reliable signal. However, these models successfully avoided a sharp market drop.

## Rationale
Futures traders are often considered "smart money" due to their access to superior market insights. This study explores whether publicly available data from the top 20 futures traders (updated daily across seven categories) can enhance ETF trading strategies. By integrating futures contract data with ETF and market index historical data, the models aim to capture predictive signals not fully reflected in ETF price movements. Backtesting validates the robustness of these strategies, providing a foundation for profitable and risk-adjusted trading.

## Research Question
Can predictive models, trained on historical ETF data, futures contract data, and top 20 trader information (e.g., trade volume, buy/sell volume), generate profitable and risk-adjusted trading strategies for index ETFs?

## Data Sources
- **CSI Index Futures**: Daily data from the China Financial Futures Exchange ([CFFEX](http://www.cffex.com.cn)), including top 20 traders by volume and other metrics.
- **CSI Index Historical Data**: Sourced from Yahoo Finance.
- **ETFs and Futures Studied**:
  | Index Futures | ETF        | Market Index       |
  |---------------|------------|--------------------|
  | IH            | 510050.SS  | CSI 50 Index       |
  | IF            | 000300.SS  | CSI 300 Index      |
  | IC            | 510500.SS  | CSI 500 Index      |
  | IM            | 512100.SS  | CSI 1000 Index     |

### Data Formats
- **Index Futures Top Trader Data**:
  ```
  | Contract | Trader   | Volume | VolChange | BuyVol | BuyVolChange | SellVol | SellVolChange | NetVolDiff | Date       |
  |----------|----------|--------|-----------|--------|--------------|---------|---------------|------------|------------|
  | IC2001   | Dealer1  | 15300  | 5392      | 5281   | 68331        | 100     | 75083         | 2008       | 2020-01-02 |
  | IC2002   | Dealer2  | 15288  | 5373      | 5268   | 2172         | 200     | 3898          | 592        | 2020-01-02 |
  ```
- **Index Futures Contract Data**:
  ```
  | Contract | Open   | High   | Low    | Volume | Amount      | OpenInterest | OpenInterestChange | Close  | SettlementPrice | LastSettlementPrice | Change1 | Change2 | Delta | Date       |
  |----------|--------|--------|--------|--------|-------------|--------------|--------------------|--------|-----------------|---------------------|---------|---------|-------|------------|
  | IC2001   | 5300.0 | 5392.4 | 5281.0 | 68331  | 7312402.38  | 75083.0      | 2008.0             | 5361.8 | 5368.8          | 5266.8              | 95.0    | 102.0   | --    | 2020-01-02 |
  | IC2002   | 5288.2 | 5373.6 | 5268.4 | 2172   | 231679.008  | 3898.0       | 592.0              | 5350.0 | 5359.2          | 5249.6              | 100.4   | 109.6   | --    | 2020-01-02 |
  ```
- **Index ETF Data**:
  ```
  | Contract | Open   | High   | Low    | Close  | Volume      | Date       |
  |----------|--------|--------|--------|--------|-------------|------------|
  | IC2001   | 5300.0 | 5392.4 | 5281.0 | 68331  | 7312402.38  | 2020-01-02 |
  | IC2002   | 5288.2 | 5373.6 | 5268.4 | 2172   | 231679.008  | 2020-01-02 |
  ```

## Methodology
### Data Preprocessing
- **Futures Trader Data**: Aggregated top 50 traders' data (contributing 97.24% of total futures volume) into one record per day, with zeros for unreported traders in any of the seven categories.
- **Feature Engineering**: Calculated technical indicators (e.g., MACD, RSI) and ETF-futures price correlations, resulting in 300+ features per day.
- **Target Variable**: Next day's percentage return (close-to-close) for the ETF.
- **Data Cleaning**: Aligned timestamps across datasets and handled missing values.

### Models Evaluated
- **Regression Models**:
  - LinearRegression, RidgeRegression, KnnRegressor, DecisionTreeRegressor, TransformedTargetRegressor, AdaBoostRegressor, XGBRegressor, VotingRegressor (ensemble of all regression models).
  - TensorFlow models (MLP, RNN, LSTM) for training and evaluation only.
- **Classification Models**:
  - LogisticRegression, KNeighborsClassifier, AdaBoostClassifier, XGBClassifier, VotingClassifier (ensemble of all classification models).
  - TensorFlow models (MLP, RNN, LSTM) for training and evaluation only.

### Model Training
- **Data Split**:
  - Training: First 85% of data.
  - Validation: Next 10% (used primarily for TensorFlow models).
  - Testing: Last 5% (unseen data).

- **Evaluation Metrics**:
  - Regression: Information Coefficient (IC%), Mean Squared Error (MSE).
  - Classification: Test accuracy.
  
- **Hyperparameter Tuning**: Conducted using cross-validation to optimize model performance.

  - GridSearchCV result for VotingRegressor:
      | GridSearchCV estimator|  Model used by estimator  |Parameter  name  used by   GridSearchCV| Parameter value found by GridSearchCV      | 
      |-------------------|-------------------------|-----------------------------------------------|-------------|
      | VotingRegressor   | LinearRegression        | LinearRegression__lr__fit_intercept           | False       | 
      | VotingRegressor   | RidgeRegression         | RidgeRegression__ridge__alpha                 | 10          | 
      | VotingRegressor   | KnnRegressor            | KnnRegressor__knn__n_neighbors                | 9           | 
      | VotingRegressor   | KnnRegressor            | KnnRegressor__knn__weights                    | 'distance'  |
      | VotingRegressor   | DecisionTreeRegressor   | DecisionTreeRegressor__dt__max_depth          | 5           | 
      | VotingRegressor   | DecisionTreeRegressor   | DecisionTreeRegressor__dt__min_samples_split  | 2           | 
      | VotingRegressor   | AdaBoostRegressor       | AdaBoostRegressor__ada__n_estimators          | 50          |           
      | VotingRegressor   | AdaBoostRegressor       | AdaBoostRegressor__ada__learning_rate         | 0.1         | 
      | VotingRegressor   | XGBRegressor            | XGBRegressor__xgb__max_depth                  | 3           |
      | VotingRegressor   | XGBRegressor            | XGBRegressor__xgb__learning_rate              | 0.01        | 
      | VotingRegressor   | XGBRegressor            | XGBRegressor__xgb__n_estimators               | 100         | 
      | VotingRegressor   |                         | weights                                       | [1, 1, 2, 1, 1, 1, 1, 1]  | 

- GridSearchCV result for VotingClassifier:
```
      | GridSearchCV estimator|Model used by estimator|  Parameter  name  used by   GridSearchCV  | Parameter value found by GridSearchCV | 
      |--------------------|--------------------------|-----------------------------------------------|-------------------|
      | VotingClassifier   | LinearRegression         | LogisticRegression__lg__fit_intercept         | True              | 
      | VotingClassifier   | KnnRegressor             | KNeighborsClassifier__knn__n_neighbors        | 7                 | 
      | VotingClassifier   | KnnRegressor             | KNeighborsClassifier__knn__weights            | 'uniform'         | 
      | VotingClassifier   | AdaBoostClassifier       | AdaBoostClassifier__ada__learning_rate        | 0.1               |
      | VotingClassifier   | AdaBoostClassifier       | AdaBoostClassifier__ada__n_estimators         | 100               | 
      | VotingClassifier   | XGBClassifier            | XGBClassifier__xgb__max_depth                 | 3                 |
      | VotingClassifier   | XGBClassifier            | XGBClassifier__xgb__learning_rate             | 0.01              | 
      | VotingClassifier   | XGBClassifier            | XGBClassifier__xgb__n_estimators              | 100               | 
      | VotingClassifier   |                          | weights                                       | [1, 1, 2, 1, 1]   | 
```
### Backtesting
- **Setup**:
  - Initial capital: $100,000.
  - Commission: 0.1%.
  - Trading rules: Buy all when prediction > 0 (regression) or True (classification); sell all when prediction < 0 or False. No shorting allowed.
  - Benchmark: Buy-and-hold strategy over the test period.
  - Trades executed at closing prices.

- **Metrics Collected**:
  - Annualized return, winning rate, number of trades, Sharpe ratio, maximum drawdown.

## Results
- **Predictive Performance**: High IC% between predicted and actual ETF returns in the test dataset, supporting further exploration of trading strategies.
- **Regression Models**: Consistently outperformed buy-and-hold strategies, with the VotingRegressor model showing significant improvements:
  - ETF 510500.SS (IC futures): **33.55%** annual return vs. **0.5%** (buy-and-hold).
  - Equal-weighted portfolio of four ETFs: **23.25%** vs. **15.15%** (buy-and-hold).
- **Classification Models**:
  - Triggered only one trade signal in a two-month test period, raising concerns about whether this is incidental or reliable. Further investigation is needed.
  - Successfully avoided a sharp market drop during the test period.
- **Performance Comparison**:
  | Index Futures | ETF        | Model Type       | Voting Model       | Annual Return (Prediction) | Annual Return (Buy-and-Hold) | Improved? |
  |---------------|------------|------------------|--------------------|----------------------------|------------------------------|-----------|
  | IH            | 510050.SS  | Regression       | VotingRegressor    | 4.43%                      | 8.51%                        | No        |
  | IF            | 000300.SS  | Regression       | VotingRegressor    | 22.50%                     | 24.50%                       | No        |
  | IC            | 510500.SS  | Regression       | VotingRegressor    | 33.55%                     | 0.50%                        | Yes       |
  | IM            | 512100.SS  | Regression       | VotingRegressor    | 32.52%                     | 27.11%                       | Yes       |
  | IH            | 510050.SS  | Classification   | VotingClassifier   | 7.42%                      | 8.51%                        | No        |
  | IF            | 000300.SS  | Classification   | VotingClassifier   | 23.87%                     | 24.50%                       | Yes       |
  | IC            | 510500.SS  | Classification   | VotingClassifier   | 0.99%                      | 0.50%                        | Yes       |
  | IM            | 512100.SS  | Classification   | VotingClassifier   | 34.13%                     | 24.50%                       | Yes       |

- **Backtesting Visuals**:
  - Regression Models: [Regression Backtrader Test Result](images/regression_backtrader_test_result.png)
  - Classification Models: [Classification Backtrader Test Result](images/classification_backtrader_test_result.png)

## Next Steps
- **Model Optimization**: Conduct extensive hyperparameter tuning, particularly for TensorFlow models (MLP, RNN, LSTM), using frameworks like KerasTuner.
- **Strategy Refinement**: Adjust trading strategies to enhance stability and reduce risk, especially for TensorFlow-based models where high IC% did not translate into effective trading outcomes.
- **Further Research**: Investigate the reliability of classification model signals, given the limited number of trades in the test period.

## Project Outline
- [Data Exploration](https://github.com/mingl2000/UCBCapstone/blob/main/UCBCapstone_data_explorer.ipynb)
- [Model Training and Testing](https://github.com/mingl2000/UCBCapstone/blob/main/UCBCapstone_models.ipynb)
- [Backtesting](https://github.com/mingl2000/UCBCapstone/blob/main/UCBCapstone_backtest.ipynb)

## Project Structure
```
UCBCapstone/
├── app_settings.py                # Application settings (random seeds, plotting)
├── README.md
├── UCBCapstone_backtest.ipynb     # Backtesting notebook
├── UCBCapstone_backtest.py        # Backtesting code
├── UCBCapstone_data_explorer.ipynb # Data exploration notebook
├── UCBCapstone_data_io.py         # Data input/output
├── UCBCapstone_data_prepare.py    # Data preprocessing (phase 2)
├── UCBCapstone_data_view.py       # Data visualization functions
├── UCBCapstone_models.ipynb       # Model training notebook
├── UCBCapstone_models.py          # Model training code
├── UCBCapstone_models_search.py   # Hyperparameter search
├── data/                          # CSV data files
├── data_prepare/                  # Data preprocessing scripts
└── images/                        # Visualization images
    ├── backtrader_test_result.png
    ├── classification_backtrader_test_result.png
    ├── regression_backtrader_test_result.png
    └── Tensorflow_MLP_training.png
```

## Contact Information
**Ming Lu**
Email: [ml124@cornell.edu](mailto:ml124@cornell.edu)  
LinkedIn: [Ming Lu](https://www.linkedin.com/in/ming-lu-4187376)
