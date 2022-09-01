import os
import pickle
import pandas as pd
import numpy as np
import datetime as dt
from dataApi.stockList import trans_windcode2int, trans_int2windcode
from dataApi.indName import sw_level1, citics_level1
from dataApi.tradeDate import get_trade_date_interval, trans_datetime2int, get_recent_trade_date, \
    get_pre_trade_date, get_date_range, trade_minutes
from BasicData.local_path import *


def load_pickle(file):

    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def get_daily_1day(factor_list, date=None, code_list=None, type='stock', base_date=20100101, diy_address=None):

    row = get_trade_date_interval(date, base_date)

    if type == 'stock':
        address = base_address + 'daily'
        if code_list == None:
            code_list = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list', start=row, stop=row+1).iloc[0]
            code_list = code_list[code_list].index.to_list()
        else:
            code_list = [trans_windcode2int(x) for x in code_list]
    elif type == 'bench':
        address = base_address + 'dailyBench'
        if code_list == None:
            code_list = ['HS300', 'ZZ500', 'ZZ800', 'SZZZ', 'SZCZ', 'ZZ1000', 'SZ50', 'ZXBZ', 'CYBZ']
    else:
        raise TypeError("type must be stock or bench")

    if diy_address != None:
        address = diy_address

    df = pd.concat([pd.read_hdf('%s/%s.h5' % (address, factor), factor, start=row, stop=row+1).iloc[0]
                    .rename(factor) for factor in factor_list], axis=1)
    if code_list != None:
        df = df.reindex(code_list)
    return df

def get_daily_1stock(code, factor_list, date_list=None, type='stock', diy_address=None):

    if type == 'stock':
        address = base_address + 'daily'
        code = trans_windcode2int(code)
    elif type == 'bench':
        address = base_address + 'dailyBench'
    else:
        raise TypeError("type must be stock or bench")

    if diy_address != None:
        address = diy_address

    df = pd.DataFrame()
    for factor in factor_list:
        temp = pd.read_hdf('%s/%s.h5' % (address, factor), factor, columns=[code])
        if len(temp.columns) == 1:
            temp.columns = [factor]
        df = pd.concat([df, temp], axis=1)

    if date_list != None:
        _date_list = [trans_datetime2int(x) for x in date_list]
        df = df.reindex(_date_list)

    return df

def get_daily_1factor(factor, date_list=None, code_list=None, type='stock', diy_address=None):

    if type == 'stock':
        address = base_address + 'daily'
        if code_list != None:
            code_list = [trans_windcode2int(x) for x in code_list]
    elif type == 'bench':
        address = base_address + 'dailyBench'
    else:
        raise TypeError("type must be stock or bench")

    if diy_address != None:
        address = diy_address

    df = pd.read_hdf('%s/%s.h5' % (address, factor), factor)
    if date_list != None:
        _date_list = [trans_datetime2int(x) for x in date_list]
        df = df.reindex(index=_date_list)
    if code_list != None:
        df = df.reindex(columns=code_list)
    return df

def get_quarter_1factor(factor, code_list=None, date_list=None):

    if code_list != None:
        code_list = [trans_int2windcode(x) for x in code_list]

    if date_list == None:
        date_list = get_date_range(20090331, None, 'R')

    date_list = [str(x) for x in date_list]

    df = fd.get_factor_value('Basic_factor', mddate=date_list, stock=code_list, factor_names=[factor]).iloc[:, 0].unstack()
    df.index = df.index.map(int)
    df.columns = df.columns.map(trans_windcode2int)
    return df

def get_single_quarter(factor, code_list=None, date_list=None):

    if code_list != None:
        code_list = [trans_int2windcode(x) for x in code_list]

    df = get_quarter_1factor(factor, code_list=code_list)
    df_sig = df.diff()
    df_sig[::4] = df[::4]
    if date_list != None:
        df_sig = df_sig.reindex(date_list)
    return df_sig

def get_ttm_quarter(factor, code_list=None, date_list=None):

    if code_list != None:
        code_list = [trans_int2windcode(x) for x in code_list]

    df = get_single_quarter(factor, code_list=code_list)
    df_ttm = df.rolling(4).sum().dropna(how='all')
    if date_list != None:
        df_ttm = df_ttm.reindex(date_list)
    return df_ttm

def get_qoq(df):

    df_qoq = df.diff() / df.shift().abs().replace(0, np.nan)
    df_qoq = df_qoq.dropna(how='all')
    return df_qoq

def get_yoy(df):

    df_yoy = df.diff(4) / df.shift(4).abs().replace(0, np.nan)
    df_yoy = df_yoy.dropna(how='all')
    return df_yoy

def _trans_financial2fixed_date(date):

    date = trans_datetime2int(date)
    year, md = divmod(date, 10000)
    if md == 331:
        _md = 430
    elif md == 630:
        _md = 831
    elif md == 930:
        _md = 1031
    elif md == 1231:
        _md = 430
        year += 1
    else:
        raise ValueError("date must be financial report date")
    return int(year * 10000 + _md)

def fill_quarter2daily_by_fixed_date(df):

    _df = df.copy().sort_index()
    _df.index = _df.index.map(_trans_financial2fixed_date)
    _df = _df[~_df.index.duplicated(keep='last')].replace(np.nan, np.inf)
    _df = _df.reindex(get_date_range(_df.index[0])).ffill().replace(np.inf, np.nan)
    return _df

def fill_quarter2daily_by_issue_date(df):

    _df = df.copy()
    _df.index = df.index.map(trans_datetime2int).map(str)
    _df.columns = df.columns.map(trans_int2windcode)
    _df.index.name = 'mddate'
    _df.columns.name = 'stock'
    issue_date = fd.get_factor_value('Basic_factor', factor_names=['stm_issuingdate'], mddate=_df.index.to_list(),
                                     stock=_df.columns.to_list()).iloc[:, 0].dropna().map(int).rename('date')
    _df = pd.concat([_df.stack(), issue_date], axis=1).reset_index()
    _df.loc[_df['date'].isnull(), 'date'] = _df.loc[_df['date'].isnull(), 'mddate'].map(_trans_financial2fixed_date)
    _df['date'] = _df['date'].map(int)
    _df = _df.loc[_df['stock'].map(lambda x: x[0] in ('0', '3', '6'))]
    _df['stock'] = _df['stock'].map(trans_windcode2int)
    _df = _df.rename(columns={'stock': 'code'})
    _df = _df.sort_values(['code', 'mddate']).drop_duplicates(subset=['code', 'date'], keep='last')
    _df = _df.pivot('date', 'code', 0)
    _df = _df.reindex(get_date_range(_df.index[0])).ffill()
    return _df

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