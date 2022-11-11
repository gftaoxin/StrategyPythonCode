import pandas as pd
import numpy as np
from BarraFactor.barra_factor import *
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import requests,json,datetime,time,sys
from dataApi import getData,tradeDate,stockList
from BasicData.local_path import *
from usefulTools import *

# 1、行业内创新高和创新低的个股数量
# 如果超过一定比例，那么认为该行业出现缝隙那；但如果超过比例的数量太多，则认为是大会而不是大风险
def high_low_stock(start_date,end_date,period='M',ind='SW1'):
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date, period=period)
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())

    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)

    # 个股所属行业
    close_badj = get_daily_1factor('close_badj', date_list=date_list)
    price_max =  close_badj == close_badj.rolling(252).max()
    price_min = close_badj == close_badj.rolling(252).min()

    up_num = pd.DataFrame(index=period_date_list, columns=ind_name)
    down_num = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in tqdm(ind_name):
        for j in range(1,len(period_date_list)):
            start,end = period_date_list[j-1], period_date_list[j]
            up_num.loc[end,i] = price_max.loc[start:end,code_ind.loc[end][code_ind.loc[end] ==i].index].max().sum() / len(code_ind.loc[end][code_ind.loc[end] ==i])
            down_num.loc[end, i] = price_min.loc[start:end,code_ind.loc[end][code_ind.loc[end] == i].index].max().sum() / len(code_ind.loc[end][code_ind.loc[end] == i])

    ind_factor = up_num[ind_useful].loc[start_date:end_date].astype(float)
    up_factor = pd.DataFrame(index = ind_factor.index,columns=ind_factor.columns)
    down_factor = pd.DataFrame(index = ind_factor.index,columns=ind_factor.columns)
    for date in ind_factor.index:
        x = ind_factor.loc[date].astype(float)
        median = np.nanmedian(x)
        mad = np.nanmedian(abs(x - median))
        high = median + 3 * 1.4826 * mad
        low = median - 1.4826 * mad

        up_factor.loc[date] = (ind_factor.loc[date] > high)
        down_factor.loc[date] = (ind_factor.loc[date] < low)

    # 如果up_factor数量较多，代表着市场要上涨



