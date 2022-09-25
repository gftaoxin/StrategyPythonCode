import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from SectorRotation.FactorTest import *

def Factor_IndayMomentum(start_date,end_date,ind):

    date_list = get_date_range(get_pre_trade_date(start_date,offset=120),end_date)
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())

    ind_close =get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_open = get_daily_1factor('open', date_list=date_list, code_list=ind_name, type=ind[:-1])

    ind_factor = (ind_close / ind_open -1 ).rolling(10).sum()

    overnight_factor = (ind_open / ind_close.shift(1)).rolling(20).apply(lambda x: x.prod())

    ind_factor = ind_factor - overnight_factor
    # ind_factor = ind_close.pct_change(20)

    return ind_factor.loc[get_pre_trade_date(start_date,offset=1):end_date]


start_date, end_date,ind = 20150101,20211130,'SW1'
factor = Factor_IndayMomentum(start_date,end_date,ind)
read_path = 'E:/FactorTest/'
factor_name = 'Factor_IndayMomentum'
factor.to_pickle(read_path+factor_name+'.pkl')

test_result, value_result = factor_in_box(factor_name, read_path,ind)
