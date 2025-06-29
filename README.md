# UCBCapstone

## Introduction
    - It's said future contract traders are smart traders. So can public future contact trading (top 20 in each of 7 columns only daily) help?
    - This study intends to investigate if this improves trading outcome of CSI index only.

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

## Dealer analysis
    - Top 50 dealers contributes 97+% trading volume
    - Top 100 dealers contributes 99+% trading volume
    - Top 50 dealers data is used for futhre analysis

## IC analysis:
    - Information Coefficient (IC) is analyed between future contract trading data and CSI index change next day
        - By each future contract trading data colum for all dealers:
            - volume        | contract volume at the end day     
            - volchange     | contract volume change between the day and the day before
            - buyvol        | buy contract volume at the end day      
            - buyvolchange  | buy contract volume change between the day and the day before  
            - sellvol       | sell contract volume at the end day 
            - sellvolchange | sell contract volume change between the day and the day before  
            - net_vol_diff  | buy volume change - sell volume change
        - By each dealer/future contract trading data column
            - IC for 2 biggest dealer is about 0.1
            - One dealer has IC between trading volume and CSI index change is 0.3. But the trading volume is small and data is not availale every day.


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

    IC from model predictions and y_test for IF future index
        - IC =-0.230 is significently higher than any of IC computed from raw future trading data and CSI index change next day
            - IC is almost the same from the Gridsearched model.
        - This is significent higher than 2 biggest deals with IC about 0.1

    IC from model predictions and y_test for all 4 future index:
        - Here is the result:
          product  | predicted_IC   | Index ETF |   Index name
            IF     | -0.238435      | 000300.SS |   CSI 300 index
            IH     |  0.134242      | 510050.SS |   CSI 50 index
            IC     |  0.024866      | 510500.SS |   CSI 500 index
            IM     | -0.257277      | 512100.SS |   CSI 1000 index
        - Based on this analysis:
            - CSI future index IF and IM may have more prediciton power for the crossponding index ETF change percent next day
                - Future study will focus on this with the hope of best trading return
            - CSI future index IC may NOT have more prediciton power for the crossponding index ETF change percent next day
            - CSI future index IC may have fairely good prediciton power for the crossponding index ETF change percent next day
            

## Study conclusion
    - GridSearchCV model got the smallest MSE, even though this MSE is only 10% smaller than the max MSE of all the models tried and IC from prediction does not improve. 
    - All the MSE is very small than expected. More study is needed.
        - The likely cause for small MSE is : The target is the change for the next day which is small, usually around -0.01 to 0.01 and fairely close with each other most of the time.
    - Dealer and correlationsof their trading to CSI 300 index percent change next day:
        - Some small dealers juding by trading volume seem to have high correlations between the two. But they may not have future trade data every day since only top 20 in each category is published.
            - This study focus on the top 50 dealers by volume.
            - Some of the dealers' net future contract trade volume is negatively correlated to the CSI 300 daily return next day. This indicates that future contact may be used to reduce trading risk on stocks instead of profiting for future contract trades.   
    - The Information Coefficient (IC) from the predicted value of VotingRegressor is high for index future product IF and IM
            - This may be used to improve the trading performance for trade related index ETF  000300.SS and 512100.SS


### Future study:
    - Add more models, such as deep neural network, etc.
    - Add back testing to see if the high IC from model predictions can actually help for the trading outcome
        - Buy and hold strategy for CSI 300 is the baseline
        - AI strategt from different models
    - Add more data sources
        - Select the top n stocks will have high ICs between future contract data of the day and stock return next day.
        - Will this strategy help to achive better return than CSI trading directly?