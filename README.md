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

* **Following models are tried and compared** and **LinearRegression** is used as **baseline model**.
  - LinearRegression
  - KnnRegressor
  - DecisionTreeRegressor
  - SupportVectorRegressor
  - TransformedTargetRegressor
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

* **Test result**:
  Here is the test summary across multiple models and mutliple csi index future products.
    - For some index future products, such as IC, the Information Coefficient% seem to be higher acoss models. 
    - Please notem the **Information Coefficient% is 100 times of Information Coefficient**.
    - Generally speaking, the RMSE is fairely consistent between models and across products with exception XgbBoost for IM.
      - More turning for XgbBoost may be needed.



| Product | Model                     | Information Coefficient%        | RMSE     | Index ETF | Index Name      |
|---------|---------------------------|------------|----------|-----------|-----------------|
| IH      | LinearRegression          | -6.956484  | 1.518327 | 510050.SS | CSI 50 Index    |
| IH      | RidgeRegression           | -7.049813  | 1.510930 | 510050.SS | CSI 50 Index    |
| IH      | KnnRegressor              | -2.127591  | 1.255726 | 510050.SS | CSI 50 Index    |
| IH      | DecisionTreeRegressor     | 3.738916   | 1.626148 | 510050.SS | CSI 50 Index    |
| IH      | SupportVectorRegressor    | -3.874771  | 1.203092 | 510050.SS | CSI 50 Index    |
| IH      | TransformedTargetRegressor| -6.956484  | 1.518327 | 510050.SS | CSI 50 Index    |
| IH      | VotingRegressor           | -4.786681  | 1.321331 | 510050.SS | CSI 50 Index    |
| IH      | AdaBoostRegressor         | -0.596053  | 1.232800 | 510050.SS | CSI 50 Index    |
| IH      | XgbBoost                  | -2.440734  | 1.235854 | 510050.SS | CSI 50 Index    |
| IH      | Tensorflow                | -17.115662 | 1.505732 | 510050.SS | CSI 50 Index    |
| IF      | LinearRegression          | -5.124202  | 1.521364 | 000300.SS | CSI 300 Index   |
| IF      | RidgeRegression           | -4.766733  | 1.505695 | 000300.SS | CSI 300 Index   |
| IF      | KnnRegressor              | -11.842341 | 1.209721 | 000300.SS | CSI 300 Index   |
| IF      | DecisionTreeRegressor     | 8.196717   | 1.641102 | 000300.SS | CSI 300 Index   |
| IF      | SupportVectorRegressor    | 0.814373   | 1.091704 | 000300.SS | CSI 300 Index   |
| IF      | TransformedTargetRegressor| -5.124202  | 1.521364 | 000300.SS | CSI 300 Index   |
| IF      | VotingRegressor           | -2.628456  | 1.245353 | 000300.SS | CSI 300 Index   |
| IF      | AdaBoostRegressor         | -6.799853  | 1.137156 | 000300.SS | CSI 300 Index   |
| IF      | XgbBoost                  | 5.392898   | 1.102974 | 000300.SS | CSI 300 Index   |
| IF      | Tensorflow                | -0.154221  | 1.393530 | 000300.SS | CSI 300 Index   |
| IC      | LinearRegression          | 27.318590  | 1.375402 | 510500.SS | CSI 500 Index   |
| IC      | RidgeRegression           | 26.856829  | 1.356191 | 510500.SS | CSI 500 Index   |
| IC      | KnnRegressor              | 1.723745   | 1.259227 | 510500.SS | CSI 500 Index   |
| IC      | DecisionTreeRegressor     | 12.690832  | 1.442431 | 510500.SS | CSI 500 Index   |
| IC      | SupportVectorRegressor    | 29.304191  | 1.122285 | 510500.SS | CSI 500 Index   |
| IC      | TransformedTargetRegressor| 27.318590  | 1.375402 | 510500.SS | CSI 500 Index   |
| IC      | VotingRegressor           | 28.317904  | 1.169722 | 510500.SS | CSI 500 Index   |
| IC      | AdaBoostRegressor         | 20.265594  | 1.165397 | 510500.SS | CSI 500 Index   |
| IC      | XgbBoost                  | 21.007844  | 1.156693 | 510500.SS | CSI 500 Index   |
| IC      | Tensorflow                | 10.461168  | 1.310815 | 510500.SS | CSI 500 Index   |
| IM      | LinearRegression          | 1.857966   | 7.008058 | 512100.SS | CSI 1000 Index  |
| IM      | RidgeRegression           | 1.610533   | 6.229373 | 512100.SS | CSI 1000 Index  |
| IM      | KnnRegressor              | 3.638734   | 7.429998 | 512100.SS | CSI 1000 Index  |
| IM      | DecisionTreeRegressor     | 16.741148  | 2.079649 | 512100.SS | CSI 1000 Index  |
| IM      | SupportVectorRegressor    | 4.500820   | 1.740467 | 512100.SS | CSI 1000 Index  |
| IM      | TransformedTargetRegressor| 1.857966   | 7.008058 | 512100.SS | CSI 1000 Index  |
| IM      | VotingRegressor           | 3.824080   | 4.124704 | 512100.SS | CSI 1000 Index  |
| IM      | AdaBoostRegressor         | 10.868526  | 1.712199 | 512100.SS | CSI 1000 Index  |
| IM      | XgbBoost                  | -2.729462  | 23.525207| 512100.SS | CSI 1000 Index  |
| IM      | Tensorflow                | -5.702529  | 1.860019 | 512100.SS | CSI 1000 Index  |




* **Insights**:
  * **LinearRegression** does not offer best RMSE in general across all products.
    - But for some products, like **IC**, information coefficient from LinearRegression is very impressive to be 0.27 or 27% reported above.

  * **Tensorflow** model:
    - It seems to be strange that the mighty Tensorflow did not offer best RMSE nor information coefficient:
      - The data may be linear by nature.
      - The Tensorflow model may be too complicated.
      - Different Tensorflow model may be needed where Keras turning may help.

  * For index future product **IC**, the information coefficient is consistently high across most of models, except KnnRegressor and Tensorflow
    - This indicates that there is a strong correlation between the CSI index change next day and this future product trading info.
    - This product may offer best insight for CSI index trading. 

  * **IH**,**IF** and **IM** futures show less predictive power for their corresponding ETF returns.
    - When Information Coefficient (IC) is **positive**, when prediction is **positive**, **positive** return next day is expected. 



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


