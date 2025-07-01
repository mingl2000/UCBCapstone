# UCB Capstone for AI/ML class

## Introduction

* It's often said that futures traders are the "smart money." This study explores whether the public data of top 20 futures contract traders daily (across 7 columns, updated daily) can be leveraged to improve trading outcomes.
* Specifically, we investigate if this data can enhance predictions of the CSI index performance.

## Jupyter Notebook

You can find the full Jupyter notebook for this study here:
👉 [data\_explore3.ipynb](https://github.com/mingl2000/UCBCapstone/blob/main/data_explore3.ipynb)

## Data Sources

* **CSI 300 Index Futures**: From [CFFEX](http://www.cffex.com.cn), including daily top 20 traders by volume and other metrics.
* **CSI 300 Index Historical Data**: From Yahoo Finance.

## Data Preprocessing

* Futures data from **January 4, 2021** onwards was collected.
* **Top 50 dealers** were selected based on total trading volume over the entire period.
* Each day's net long/short positions for these 50 dealers were organized into a single row. If a dealer has no data for a specific day, a value of 0 was used.
* The futures data was merged with the CSI index daily return data to form a dataset like:

```
| Date       | Dealer1 | Dealer2 | Dealer3 | CSI return (next day) |
|------------|---------|---------|---------|------------------------|
| 2025-01-02 |  -100   |   80    |  -70    |         -0.01         |
| 2025-01-03 |   200   |  -100   |   70    |         -0.02         |
```

## Dealer Analysis

* Top 50 dealers account for over **97%** of the total trading volume.
* Top 100 dealers account for over **99%**.
* Therefore, the **top 50 dealers** are used for all further analysis.

## Information Coefficient (IC) Analysis

We analyzed the **Information Coefficient** between futures contract trading data and next-day CSI index returns.

* **Per data column (aggregated across dealers):**

  * `volume`: End-of-day contract volume
  * `volchange`: Daily change in contract volume
  * `buyvol`: End-of-day buy contract volume
  * `buyvolchange`: Daily change in buy contract volume
  * `sellvol`: End-of-day sell contract volume
  * `sellvolchange`: Daily change in sell contract volume
  * `net_vol_diff`: `buyvolchange - sellvolchange`

* **Per dealer:**

  * The two largest dealers showed IC ≈ **0.1**
  * One smaller dealer had an IC ≈ **0.3**, but their data was not consistently available and their trading volume was small.

## Models & Performance

| Model                        | MSE      | Comments                                  |
| ---------------------------- | -------- | ----------------------------------------- |
| Linear Regression            | 0.000138 | Default parameters                        |
| Ridge Regression             | 0.000138 | Default parameters                        |
| KNN Regressor                | 0.000124 | Default parameters                        |
| Decision Tree Regressor      | 0.000251 | Default parameters                        |
| Support Vector Regressor     | 0.000169 | Default parameters                        |
| Transformed Target Regressor | 0.000138 | Default parameters                        |
| Voting Regressor             | 0.000129 | Ensemble of above models                  |
| GridSearchCV                 | 0.000123 | Hyperparameter tuning on all above models |


### IC from Model Predictions vs. Actual CSI Index Return (Next Day)

* **IF index**:

  * IC = **-0.238**, which is significantly better than raw ICs from futures data
  * Grid-searched model achieves similar IC

* **Across 4 Futures Products**:

| Product | Information Coefficient (IC) from  **VotingRegressor**  | Information Coefficient (IC) from  **AdaBoostRegressor**    |     Index ETF   | Index Name      |  
|---------|---------------------------------------------------------|-------------------------------------------------------------|-----------------|-----------------|
| IF      |           -0.238                                        |           -0.116083                                         |     000300.SS   | CSI 300 Index   |  
| IH      |            0.134                                        |            0.103926                                         |     510050.SS   | CSI 50 Index    |  
| IC      |            0.025                                        |           -0.222088                                         |     510500.SS   | CSI 500 Index   |  
| IM      |           -0.257                                        |            0.102572                                         |     512100.SS   | CSI 1000 Index  |  




* **Insights**:

  * **IF** and **IM** futures show strong predictive power for their corresponding ETF returns.
    - Since Information Coefficient (IC) is **negative**, when prediction is **positive**, **negative** return next day is expected. 
  * **IC** appears to have limited predictive power from VotingRegressor, but **AdaBoostRegressor** seems to provide good IC
  * **IH** shows moderate predictive potential
    - Since Information Coefficient (IC) is **positive**, when prediction is **positive**, **positive** return next day is expected. 


### Try with AdaBoostRegressor for all CSI future index products
    - Information Coefficient (IC) for product IC got -0.22 which is much more significent than the 0.02 from Regressor

## Study Conclusions

* The **GridSearchCV model** achieved the lowest MSE, though the improvement over other models was minor (\~10%).
* All MSE values were much lower than expected, likely due to the small variance in next-day index changes (typically between -0.01 and 0.01).
* While large dealers show modest correlation with CSI returns, some smaller dealers (with intermittent data) showed higher correlations.

  * This suggests futures contracts may often be used for **hedging** rather than speculation.
* The **VotingRegressor** model yielded a high absoute value of Information Coefficient (IC) for **IF** and **IM** futures:
  * This could be leveraged to enhance ETF trading performance (e.g., 000300.SS and 512100.SS).

* The **AdaBoostRegressor** model yielded a high absoute value of Information Coefficient (IC)  for **IC** future:
  * This could be leveraged to enhance ETF trading performance (e.g., 510500.SS).
*

## Future Work

* **Add more models**, such as deep learning approaches (e.g., LSTM, Transformers).
* **Backtesting** strategies based on model predictions:

  * Compare against a simple "buy and hold" strategy for CSI 300
  * Evaluate whether AI-based strategies provide better returns
* **Expand data sources**:

  * Identify top `n` individual stocks where futures data correlates highly with stock return the next day
  * Investigate whether this selective strategy outperforms direct index-based trading


