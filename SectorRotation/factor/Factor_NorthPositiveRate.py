import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from SectorRotation.FactorTest import *

def Factor_NorthPositiveRate(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=200), end_date)
    north_volume = get_daily_1factor('north_quantity', date_list=date_list).dropna(how='all')
    close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
    north_amt = (north_volume.diff(1) * close).dropna(how='all')

    # 个股所属行业
    code_ind = get_daily_1factor(ind, date_list=date_list)
    amt = get_daily_1factor('amt', date_list=date_list)
    free_mv = get_daily_1factor('mkt_free_cap', date_list=date_list)
    # 因子1：北向资金行业累计净流入金额/行业在北向资金内个股的累计成交额
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())

    north_ind_mv = pd.concat([free_mv[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1).dropna(how='all')
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)

    north_ind_positive_in = north_ind_amt_in.copy()
    north_ind_positive_in[north_ind_positive_in < 0] = 0

    factor = north_ind_amt_in.rolling(120).sum() / north_ind_positive_in.rolling(120).sum()
    ind_df = (factor).dropna(how='all')

    return ind_df.loc[get_pre_trade_date(start_date, offset=1):end_date]

start_date, end_date,ind = 20170101,20211130,'SW1'
factor = Factor_NorthPositiveRate(start_date,end_date,ind)
factor_name = 'Factor_NorthPositiveRate'

test_result, value_result = factor_in_box(factor,factor_name, ind,fee=0)

self = FactorTest(test_start_date=20150101, test_end_date=20211130, ind=ind, day=20)
test_factor = self.deal_factor(factor).rank(ascending=False,axis=1) <= 5
result, all_net_value,excess_value = self.strategy_test(factor, fee=0.004, save_path='E:/FactorTest/',save_name = 'strategy')





