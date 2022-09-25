import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from SectorRotation.FactorTest import *

def Factor_Dragon(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date,offset=120),end_date)
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())
    # 个股所属行业
    code_ind = get_daily_1factor(ind, date_list=date_list)
    amt = get_daily_1factor('amt', date_list=date_list)
    amt_20days = amt.rolling(20).mean()

    close_badj = get_daily_1factor('close_badj', date_list=date_list)
    pct_20days = close_badj.pct_change(20)
    # 因子：
    # 龙头股：行业内最近20日成交金额占比前60%的成分股
    # 普通股：其余个股
    # 分歧度因子：龙头股20日收益率 - 普通股20日收益率
    dragon_factor = pd.DataFrame(index=date_list,columns=ind_name)

    for i in ind_name:
        dragon_pct = pct_20days[amt_20days[code_ind == i].rank(pct=True) >=0.4].mean(axis=1)
        other_pct = pct_20days[amt_20days[code_ind == i].rank(pct=True) <0.4].mean(axis=1)

        dragon_factor[i] = (dragon_pct - other_pct)

    dragon_factor =  (dragon_factor.sub(dragon_factor.mean(axis=1),axis=0).div(dragon_factor.std(axis=1),axis=0)) ** 2

    return dragon_factor.loc[get_pre_trade_date(start_date,offset=1):end_date]


start_date, end_date,ind = 20150101,20211130,'SW1'
factor = Factor_Dragon(start_date,end_date,ind)
factor_name = 'Factor_Dragon'

test_result, value_result = factor_in_box(factor,factor_name, ind,fee=0)



