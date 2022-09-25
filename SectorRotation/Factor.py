import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from SectorRotation.FactorTest import *

def Factor_NorthMoneyIn(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=200), end_date)
    north_volume = get_daily_1factor('north_quantity', date_list=date_list).dropna(how='all')
    close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
    north_amt = (abs(north_volume.diff(1)) * close).dropna(how='all')

    # 个股所属行业
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_mv = get_modified_ind_mv(date_list=date_list,ind_type = ind)
    # 使用收益率和市值做回归取残差，滚动12个月
    ind_close = get_daily_1factor('close', date_list=date_list, code_list=ind_name, type=ind[:-1])
    ind_pct = ind_close.pct_change(20)

    for industry in ind_name:
        ind_pct[industry]
        ind_mv[industry]



    a.columns = pd.Series(a.columns).apply(lambda x:ind_name[x])


    # 因子1：北向资金行业累计净流入金额/行业在北向资金内个股的累计成交额
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)

    factor = north_ind_amt_in.rolling(60).sum()
    ind_df = (factor).dropna(how='all')

    return ind_df.loc[get_pre_trade_date(start_date, offset=1):end_date]


start_date, end_date,ind = 20170101,20211130,'SW1'
factor = Factor_NorthMoneyIn(start_date,end_date,ind)
read_path = 'E:/FactorTest/'
factor_name = 'Factor_NorthMoneyIn'
factor.to_pickle(read_path+factor_name+'.pkl')

test_result, value_result = factor_in_box(factor_name, read_path,ind)

