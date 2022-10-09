import time,os,matplotlib,warnings
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
import numba
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 其余
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
    elif ind_type == 'CITICS':
        ind = get_daily_1factor('CITICS1', date_list, code_list).values.T
    ind_codes = get_ind_con(ind_type, level=1)

    for i in ind_codes:
        exist = ind == i
        np.subtract(arr, np.nanmedian(np.ma.array(arr, mask=~exist).data, axis=0), out=arr, where=exist)
        np.divide(arr, np.nanmedian(np.abs(np.ma.array(arr, mask=~exist).data), axis=0), out=arr, where=exist)

    return df

# 函数1：获取时序分位数
def rollingRankArgSort(array):
    return (array.argsort().argsort() + 1)[-1]/ len(array)

def ts_rank(df, rol_day='history'):
    if rol_day == 'history':
        df = df.expanding().apply(lambda x: rollingRankArgSort(x.values))
    else:
        df = df.rolling(rol_day, min_periods=60).apply(lambda x: rollingRankArgSort(x.values))
    return df

# 函数2：获取两个dataframe相关系数
def array_coef(x, y):
    x_values = np.array(x, dtype=float)
    y_values = np.array(y, dtype=float)
    x_values[np.isinf(x_values)] = np.nan
    y_values[np.isinf(y_values)] = np.nan
    nan_index = np.isnan(x_values) | np.isnan(y_values)
    x_values[nan_index] = np.nan
    y_values[nan_index] = np.nan
    delta_x = x_values - np.nanmean(x_values, axis=0)
    delta_y = y_values - np.nanmean(y_values, axis=0)
    multi = np.nanmean(delta_x * delta_y, axis=0) / (np.nanstd(delta_x, axis=0) * np.nanstd(delta_y, axis=0))
    multi[np.isinf(multi)] = np.nan
    return pd.Series(multi, index=x.columns)

def rolling_corr(df_x, df_y, window=None):
    """"""
    assert df_x.shape[0] == df_y.shape[0], 'dims must be same'

    corr = pd.DataFrame(np.nan, index=df_x.index, columns=df_x.columns)

    if window == None or window <= 0:
        window = df_x.shape[0]
    if window <= df_x.shape[0] and window > 1:
        for idx, index in enumerate(df_x.index):
            if idx >= window - 1:
                corr.loc[index] = array_coef(df_x.iloc[idx - window + 1:idx + 1],
                                             df_y.iloc[idx - window + 1:idx + 1]).values
    return corr

# 函数3：获取累计N日为True
def ContinuousTrueTime(df):
    if type(df) == pd.core.series.Series:
        df = pd.DataFrame(df.rename(0))
        df = df.cumsum() - df.cumsum()[df == 0].ffill().fillna(0)
        df.fillna(0, inplace=True)
        return df[0]
    else:
        df = df.cumsum() - df.cumsum()[df == 0].ffill().fillna(0)
        df.fillna(0, inplace=True)

        return df






