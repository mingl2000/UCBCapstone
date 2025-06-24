# UCBCapstone

## Jupyter notebook for this study:
    https://github.com/mingl2000/UCBCapstone/blob/main/data_explore3.ipynb

## Data source:
    CSI 300 index future data from http://www.cffex.com.cn  where daily dump of top 20 dealers by volume, etc.
    Yahoo data for CSI 300 index daily history

## Data Preprocessing:
    - All the future data from 1/4/2021 were downloaded.
    - Top 50 dealers were selected based on the total trading volume from day 1.
    - Each day's net long/short contacts for each of these 50 traders are organized into one row of data.  If no data for a dealer for that day, 0 is used.
    - This data is merged  merged CSI daily history data into data like following:

        | Date             | Dealerr1 | Dealer2 |Dealer 3 |CSI return next day
        | 2025-01-02       |   -100    | 80     |  -70    |   -0.01
        | 2025-01-03       |   200     | -100   |  70     |   -0.02
## Data models:
    The following models are tried and their results
                Model                   |   MSE             |   Comments
1       |     LinearRegression          |   0.000138        |   Default parameters
2       |      RidgeRegression          |   0.000138        |   Default parameters
3       |         KnnRegressor          |   0.000124        |   Default parameters
4       |    DecisionTreeRegressor      |   0.000251        |   Default parameters
5       |   SupportVectorRegressor      |   0.000169        |   Default parameters
6       |  TransformedTargetRegressor   |   0.000138        |   Default parameters
7       |      VotingRegressor          |   0.000129        |   Default parameters with each of the 6 models above in this table
8       |         GridSearchCV          |   0.000123        |   Hyper-parameter search for each of the 7 models above.

## Study conclusion
    - GridSearchCV model got the smallest MSE, even though this MSE is only 10% smaller than the max MSE of all the models tried.
    - All the MSE is very small than expected. More study is needed.
        - The likely cause for small MSE is : The target is the change for the next day which is small, usually around -0.01 to 0.01 and fairely close with each other most of the time.
    - Juding by the correlation between net long/short contract volume on a day and the next day's CSI 300 daily return, some of the dealers out of the 50 largest dealers seem to have to higher absolute correlation which indicates that these dealer's future trading data may be ussed to improve the timming of trades.
       - Some of the dealers' net future contract trade volume is negatively correlated to the CSI 300 daily return next day. This indicates that future contact may be used to reduce trading risk on stocks instead of profiting for future contract trades.   

