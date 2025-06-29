import os
import sys
import numpy as np
import pandas as pd
import chardet
import glob
dealer_names= ['中信期货','国泰君安','中信建投']
#future_names =['IM']
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']
def rename_cols(df, dealer):
    new_df = df[df.index == dealer]
    new_df = new_df.reset_index(drop=True)
    new_cols={}
    for col in new_df.columns:
        if col =='交易日' or col =='合约':
            pass
        else:
            new_cols[col]=col+"_" + (dealer.replace('(代客)',''))
    return new_df.rename(columns=new_cols)
    

def extract_data(filename, dealer_names, df):
    try:
        df_file = pd.read_csv(filename, encoding="gb2312")
        print(filename)
        print(df_file.head(2))
        df2 =df_file.copy()
        df2.reset_index() 
        df2.columns =['交易日','合约','排名','成交量会员简称','成交量','成交量增减','买单量会员简称','持买单量','买单量增减', '卖单量会员简称','持卖单量','卖单量增减']
        df2= df2.iloc[1:,:]
        for col in ['成交量','成交量增减','持买单量','买单量增减','持卖单量','卖单量增减']:
            df2[col]=df2[col].astype(int) 


        df2['成交量会员'] = df2['成交量会员简称'].str.replace('(代客)','')
        df2['买单量会员'] = df2['买单量会员简称'].str.replace('(代客)','')
        df2['卖单量会员'] = df2['卖单量会员简称'].str.replace('(代客)','')
        df2['成交量会员'] = df2['成交量会员'].str.replace('(经纪)','')
        df2['买单量会员'] = df2['买单量会员'].str.replace('(经纪)','')
        df2['卖单量会员'] = df2['卖单量会员'].str.replace('(经纪)','')

        trade_vol_by_dealer = df2[['成交量会员', '成交量','成交量增减']].groupby('成交量会员').sum()
        buy_vol_by_dealer = df2[['买单量会员', '持买单量','买单量增减']].groupby('买单量会员').sum()
        sell_vol_by_dealer = df2[['卖单量会员', '持卖单量','卖单量增减']].groupby('卖单量会员').sum()

        df_merged = trade_vol_by_dealer.merge(buy_vol_by_dealer, left_index=True, right_index=True, how='outer').merge(sell_vol_by_dealer, left_index=True, right_index=True, how='outer')

        df_merged =df_merged.fillna(0)
        df_merged['net_vol_diff'] =df_merged['买单量增减'].astype(int)- df_merged['卖单量增减'].astype(int)
        df_merged['datetime'] = df2['交易日'][-1:].values[0].astype(int)
        #df_merged = df_merged.reindex()
        #df_merged['dealer']=df_merged.index
        if len(df) ==0:
            df = df_merged
        else:
            df =pd.concat([df, df_merged], ignore_index=False)
        return df
    except:
        print('Failed to extract_data:', filename)
        return df
import datetime
import argparse
def main(product):
    df = pd.DataFrame()
    script_path = os.path.abspath(__file__).replace('\\', '/')
    script_directory = os.path.dirname(script_path)
    data_folder=f"{script_directory}/downloaded"
    for fn in glob.glob(os.path.join(data_folder, "*.csv")):
        filename =fn.replace('\\', '/')
        if os.path.isfile(filename):
            if os.path.getsize(filename)>0  and filename.endswith(f"_{product}_1.csv"):
                print(filename)
                df=extract_data(filename, dealer_names, df)
                print(len(df))
                
    if len(df)>0:
        df['datetime'] = df['datetime'].astype(int)
        last_trade_date= f"{df['datetime'][-1:].values[0]}"
        outfile_name = f"{script_directory}/future_{product}_{last_trade_date}.csv"
        print(df.head(1))
        df.columns=['volume' ,'volchange','buyvol', 'buyvolchange','sellvol','sellvolchange', 'net_vol_diff','datetime']
        df.to_csv(outfile_name, index=True,encoding="gb18030")
        print('data extracted into:', outfile_name)
    else:
        print(f'no valid data for {product}')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='future_summary')
    parser.add_argument('--product', type=str, required=False, default='IF', help='ticker')
    
    args = parser.parse_args()
    # Run the analysis
    main(args.product)
    