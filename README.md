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

The **Information Coefficient** between futures contract trading data and next-day CSI index returns are analyzed in addition to RMSE.

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




### IC from Model Predictions vs. Actual CSI Index Return % (Next Day)

* **Multiple models are tried**:
  - VotingRegressor using the following models:
    - LinearRegression
    - KnnRegressor
    - DecisionTreeRegressor
    - SupportVectorRegressor
    - TransformedTargetRegressor
  - AdoBoost
  - XgbBoost
  - TensorFlow 


  * IC = **-0.238**, which is significantly better than raw ICs from futures data
  * Grid-searched model achieves similar IC

* **Across 4 Futures Products**:
  Here is the test summary across multiple models and mutliple csi index future products.
    - For some index future products, such as IC, the Information Coefficient% seem to be higher acoss models. 
    - Please notem the **Information Coefficient% is 100 times of Information Coefficient**.
    - Generally speaking, the RMSE is fairely consistent between models and across products with exception XgbBoost for IM.
      - More turning for XgbBoost may be needed.

```markdown
| Product | AI Model           | Information Coefficient (IC)% | RMSE      | Index ETF  | Index Name     |
|---------|--------------------|------------------------------|-----------|------------|-----------------|
| IF      | Tensorflow         | 0.401426                     | 1.239543  | 000300.SS  | CSI 300 Index   |
| IF      | XgbBoost           | 5.392898                     | 1.102974  | 000300.SS  | CSI 300 Index   |
| IF      | VotingRegressor    | -2.628456                    | 1.245353  | 000300.SS  | CSI 300 Index   |
| IF      | AdaBoostRegressor  | -6.799853                    | 1.137156  | 000300.SS  | CSI 300 Index   |
| IH      | Tensorflow         | -9.638394                    | 1.415375  | 510050.SS  | CSI 300 Index   |
| IH      | XgbBoost           | -2.440734                    | 1.235854  | 510050.SS  | CSI 50 Index    |
| IH      | VotingRegressor    | -4.786681                    | 1.321331  | 510050.SS  | CSI 50 Index    |
| IH      | AdaBoostRegressor  | -0.596053                    | 1.232800  | 510050.SS  | CSI 50 Index    |
| IC      | Tensorflow         | 8.614573                     | 1.352025  | 510500.SS  | CSI 50 Index    |
| IC      | XgbBoost           | 21.007844                    | 1.156693  | 510500.SS  | CSI 500 Index   |
| IC      | VotingRegressor    | 28.317904                    | 1.169722  | 510500.SS  | CSI 500 Index   |
| IC      | AdaBoostRegressor  | 20.265594                    | 1.165397  | 510500.SS  | CSI 500 Index   |
| IM      | Tensorflow         | -5.642971                    | 2.442851  | 512100.SS  | CSI 1000 Index  |
| IM      | XgbBoost           | -2.729462                    | 23.525207 | 512100.SS  | CSI 1000 Index  |
| IM      | VotingRegressor    | 3.824080                     | 4.124704  | 512100.SS  | CSI 1000 Index  |
| IM      | AdaBoostRegressor  | 10.868526                    | 1.712199  | 512100.SS  | CSI 1000 Index  |
```


* **Insights**:

  * **IF** and **IM** futures show strong predictive power for their corresponding ETF returns.
    - Since Information Coefficient (IC) is **negative**, when prediction is **positive**, **negative** return next day is expected. 
  * **IC** appears to have limited predictive power from VotingRegressor, but **AdaBoostRegressor** seems to provide good IC
  * **IH** shows moderate predictive potential
    - Since Information Coefficient (IC) is **positive**, when prediction is **positive**, **positive** return next day is expected. 


### Try with AdaBoostRegressor for all CSI future index products
    - Information Coefficient (IC) for product IC got -0.22 which is much more significent than the 0.02 from Regressor

## Study Conclusions

* While large dealers show modest correlation with CSI returns, some smaller dealers (with intermittent data) showed higher correlations.

  * This suggests futures contracts may often be used for **hedging** rather than speculation.
* For CSI index future product "IC", multiple models yielded a high absoute value of Information Coefficient (IC):
  * This could be leveraged to enhance ETF trading performance (e.g., 510500.SS ).

*

## Future Work

* **Backtesting** strategies based on model predictions:
  * Compare against a simple "buy and hold" strategy for CSI 300
  * Evaluate whether AI-based strategies provide better returns
  
* **Expand data sources**:

  * Identify top `n` individual stocks where futures data correlates highly with stock return the next day
  * Investigate whether this selective strategy outperforms direct index-based trading


