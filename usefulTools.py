import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np


# 其余数据中性化处理
def get_modified_ind_mv(date_list=None, code_list=None, ind_type='SW'):

    mv = np.log(get_daily_1factor('f', date_list, code_list).values)
    if ind_type == 'SW':
        ind = get_daily_1factor('SW1', date_list, code_list).values
        ind2 = get_daily_1factor('SW2', date_list, code_list).values
        ind[ind == 6134] = ind2[ind == 6134]
        ind_codes = list(sw_level1.keys())
        ind_codes.remove(6134)
        ind_codes += [613401, 613402, 613403]
    elif ind_type == 'CITICS':
        ind = get_daily_1factor('CITICS1', date_list, code_list).values
        ind2 = get_daily_1factor('CITICS2', date_list, code_list).values
        ind[ind == 'b10m'] = ind2[ind == 'b10m']
        ind_codes = list(citics_level1.keys())
        ind_codes.remove('b10m')
        ind_codes += ['b10m01', 'b10m02', 'b10m03']
    elif isinstance(ind_type, pd.DataFrame):
        ind = ind_type.reindex(index=date_list, columns=code_list).values
        ind_codes = list(set(ind.ravel().tolist()) - {np.nan})
    else:
        raise TypeError("ind_type must be SW, CITICS or pandas.DataFrame object")
    return np.r_['0,3', tuple(ind == x for x in ind_codes)], mv

def get_reg_grow(df, window=4):

    return df.rolling(window).apply(lambda x: (np.arange(window) - (window - 1) / 2).dot(x))

def get_ts_neutral(df, window=4):

    return df / df.rolling(window).mean()

def get_ind_neutral(df, ind_type='SW'):

    date_list = [get_recent_trade_date(x) for x in df.index]
    code_list = df.columns.to_list()
    arr = df.values.T

    if ind_type == 'SW':
        ind = get_daily_1factor('SW1', date_list, code_list).values.T
        ind2 = get_daily_1factor('SW2', date_list, code_list).values.T
        ind[ind == 6134] = ind2[ind == 6134]
        ind_codes = list(sw_level1.keys())
        ind_codes.remove(6134)
        ind_codes += [613401, 613402, 613403]
    elif ind_type == 'CITICS':
        ind = get_daily_1factor('CITICS1', date_list, code_list).values.T
        ind2 = get_daily_1factor('CITICS2', date_list, code_list).values.T
        ind[ind == 'b10m'] = ind2[ind == 'b10m']
        ind_codes = list(citics_level1.keys())
        ind_codes.remove('b10m')
        ind_codes += ['b10m01', 'b10m02', 'b10m03']
    elif isinstance(ind_type, pd.DataFrame):
        ind = ind_type.reindex(index=date_list, columns=code_list).values.T
        ind_codes = list(set(ind.ravel().tolist()) - {np.nan})
    else:
        raise TypeError("ind_type must be SW, CITICS or pandas.DataFrame object")

    for i in ind_codes:
        exist = ind == i
        np.subtract(arr, np.nanmedian(np.ma.array(arr, mask=~exist).data, axis=0), out=arr, where=exist)
        np.divide(arr, np.nanmedian(np.abs(np.ma.array(arr, mask=~exist).data), axis=0), out=arr, where=exist)

    return df

if __name__ == '__main__':
    df = get_quarter_1factor('tot_oper_rev')
    df = get_ind_neutral(get_quarter_1factor('tot_oper_rev'))
