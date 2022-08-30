import time
import threading
import pandas as pd
import numpy as np
import datetime as dt
import os,pickle,gc,re
from dataApi.TradeDate import get_date_range, get_pre_trade_date, _check_input_date, get_recent_trade_date, trans_datetime2int, get_trade_date_interval
from functools import reduce
import cx_Oracle
address = 'D:\Program\BasicData\\'
con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True)

def _handle_params(trading_codes=None, date_list=None, factor_list=None):
    """
    处理股票代码、日期、因子参数的格式
    :param trading_codes: 股票代码(string)或股票列表(list)
    :param date_list: 日期(string、int)或日期列表(list)
    :param factor_list: 因子(string)或因子列表(list)
    :return: 参数字典(dict)
    """
    params_dict = {}
    # 股票代码
    if trading_codes:
        if isinstance(trading_codes, str):
            code_style = '='
            stock_codes = "'" + trading_codes + "'"
            trading_codes = [trading_codes]
            params_dict['trading_codes'] = [trading_codes, stock_codes, code_style]
        elif isinstance(trading_codes, list):
            trading_codes = list(set(trading_codes))
            if len(trading_codes) == 1:
                code_style = '='
                stock_codes = "'" + trading_codes[0] + "'"
            else:
                code_style = 'in'
                stock_codes = tuple(trading_codes)
            params_dict['trading_codes'] = [trading_codes, stock_codes, code_style]
        else:
            raise Exception("trading_codes 为股票代码(string)，或股票代码列表(list) ! ")
    # 日期
    if date_list:
        if isinstance(date_list, int):
            date_list = str(date_list)
            date_style = '='
            dates = "'" + date_list + "'"
            date_list = [date_list]
            params_dict['date_list'] = [date_list, dates, date_style]
        elif isinstance(date_list, str):
            date_style = '='
            dates = "'" + date_list + "'"
            date_list = [date_list]
            params_dict['date_list'] = [date_list, dates, date_style]
        elif isinstance(date_list, list):
            date_list = [str(date) if isinstance(date, int) else date for date in date_list]
            date_list = list(set(date_list))
            if len(date_list) == 1:
                dates = date_list[0]
                date_style = '='
            else:
                date_style = 'in'
                dates = tuple(date_list)
            params_dict['date_list'] = [date_list, dates, date_style]
        else:
            raise Exception("date_list 为单个日期(string or int)，或日期列表(list) ! ")
    # 因子
    if factor_list:
        if isinstance(factor_list, str):
            factor_list = [factor_list]
        elif isinstance(factor_list, list):
            factor_list = factor_list
            # factor_list = list(set(factor_list))
        else:
            raise Exception("factor_list 为单个因子(string)，或多个因子的列表(list) ! ")
        fields = ""
        for factor in factor_list:
            fields += factor + ','
        fields = fields[:-1]
        params_dict['factor_list'] = [factor_list, fields]

    return params_dict

# 函数1.1：股票代码和wind代码的互换
def trans_windcode2int(code):

    if isinstance(code, int) | isinstance(code, np.int64) | isinstance(code, np.int32) | isinstance(code, np.int):
        return code
    elif isinstance(code, float):
        return int(code)
    elif isinstance(code, str):
        if code in ["000001.SH", "000016.SH", "000300.SH", "000905.SH", "000906.SH",
                    "000852.SH", "399001.SZ", "399006.SZ", "399101.SZ", "399102.SZ"]:
            return int('9' + code[:-3])
        elif code.isdigit():
            return int(code)
        else:
            return int(code[:-3])
    else:
        raise Exception('input code type error')

def trans_int2windcode(code):

    if isinstance(code, str):
        return code
    elif isinstance(code, (float, int, np.int)):
        temp = str(int(code)).zfill(6)
        if temp[0] == '9' and len(temp) == 7:  # 指数
            if temp[1] == '3':
                result = temp[1:] + '.SZ'
            else:
                result = temp[1:] + '.SH'
        elif temp[0] == '0' or temp[0] == '3':
            result = temp + '.SZ'
        elif temp[0] == '6':
            result = temp + '.SH'
        else:
            result = temp + 'SH'
        return result
    else:
        raise Exception('input code type error')

# 函数2.1：获取某一日的股票列表
def get_stock_list(date=None,address = address):
    if date != None:
        date = trans_datetime2int(date)
    date = get_recent_trade_date(date)
    stock_list = pd.read_pickle(address + '/stock_list.pkl').loc[date]
    stock_list[stock_list].index.to_list()

    #row = get_trade_date_interval(date)
    #stock_list = pd.read_hdf(address + '/stock_list.h5', 'stock_list', start=row, stop=row+1).iloc[0]
    #stock_list = stock_list[stock_list].index.to_list()

    return stock_list



def _get_ind_con(date, code, ind_type='CITICS', level=1):

    _date = _check_input_date(date)

    if not isinstance(code, list):
        _code = [code]
    else:
        _code = code
    _code = [trans_int2windcode(x) for x in _code]

    df = fd.hsi(_code, _date, ind_type, level)
    df.columns = ['date', 'code', 'ind']
    df['date'] = df['date'].map(int)
    df['code'] = df['code'].map(lambda x: int(x[:6]))

    if ind_type == 'SW':
        df['ind'] = df['ind'].map(int)
    return df

def _store_ind_con(address='/data/group/800442/800319/junkData/daily'):

    date = get_date_range(20100101)
    code = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list', start=-1).columns.to_list()
    df_citics1 = _get_ind_con(date, code, ind_type='CITICS', level=1).pivot('date', 'code', 'ind').convert_objects()
    df_citics2 = _get_ind_con(date, code, ind_type='CITICS', level=2).pivot('date', 'code', 'ind').convert_objects()
    df_citics3 = _get_ind_con(date, code, ind_type='CITICS', level=3).pivot('date', 'code', 'ind').convert_objects()
    df_sw1 = _get_ind_con(date, code, ind_type='SW', level=1).pivot('date', 'code', 'ind').convert_objects()
    df_sw2 = _get_ind_con(date, code, ind_type='SW', level=2).pivot('date', 'code', 'ind').convert_objects()
    df_sw3 = _get_ind_con(date, code, ind_type='SW', level=3).pivot('date', 'code', 'ind').convert_objects()

    df_citics1.to_hdf('%s/CITICS1.h5' % address, 'CITICS1', format='t')
    df_citics2.to_hdf('%s/CITICS2.h5' % address, 'CITICS2', format='t')
    df_citics3.to_hdf('%s/CITICS3.h5' % address, 'CITICS3', format='t')
    df_sw1.to_hdf('%s/SW1.h5' % address, 'SW1', format='t')
    df_sw2.to_hdf('%s/SW2.h5' % address, 'SW2', format='t')
    df_sw3.to_hdf('%s/SW3.h5' % address, 'SW3', format='t')

def clean_stock_list(stock_list='ALL', no_ST=True, least_live_days=240, no_pause=True, least_recover_days=1,
                     no_pause_limit=0.5, no_pause_stats_days=120, no_limit_up=False, no_limit_down=False,
                     other_limit=None, start_date=None, end_date=None, trade_mode=False,
                     address='/data/group/800442/800319/junkData/daily'):

    today = get_recent_trade_date(dividing_point=8.8 if trade_mode else 17.3)

    requires = {}
    if stock_list == 'ALL':
        stock_list = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list')
    elif stock_list == 'COMMON':
        stock_list = pd.read_hdf('%s/common_stock_list.h5' % address, 'common_stock_list')
    elif stock_list in ('HS300', 'ZZ500', 'ZZ1000', 'SZ50'):
        stock_list = pd.read_hdf('%s/common_stock_list.h5' % address, stock_list)
    else:
        raise ValueError("stock_list must in (ALL, COMMON, HS300, ZZ500, ZZ1000, SZ50).")

    stock_list = stock_list.loc[:today].shift(-1) > 0.5 if trade_mode else stock_list.loc[:today] > 0.5
    requires['stock_list'] = stock_list

    if no_ST:
        ST = pd.read_hdf('%s/ST.h5' % address, 'ST').reindex_like(stock_list)
        requires['ST'] = ST != True

    if least_live_days >= 2:
        live_days = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list').cumsum()
        live_days = live_days.shift(-1) if trade_mode else live_days
        live_days = live_days.reindex_like(stock_list)
        requires['live_days'] = live_days >= least_live_days

    if no_pause:
        pause = pd.read_hdf('%s/pause.h5' % address, 'pause')
        pause = pause.shift(-1) if trade_mode else pause
        pause1 = pause[pause > 0.5].ffill(limit=least_recover_days-1) if least_recover_days >= 2 else pause[pause == True]
        requires['pause'] = pause1.isnull().reindex_like(stock_list)

        if no_pause_limit > 0:

            pause2 = pause.rolling(no_pause_stats_days, min_periods=1).sum() / pause.rolling(
                no_pause_stats_days, min_periods=1).count()

            requires['pause2'] = pause2.isnull().reindex_like(stock_list) < 0.5

    if no_limit_up:
        limit_up = pd.read_hdf('%s/limit_up.h5' % address, 'limit_up').reindex_like(stock_list)
        requires['limit_up'] = limit_up != True

    if no_limit_down:
        limit_down = pd.read_hdf('%s/limit_down.h5' % address, 'limit_down').reindex_like(stock_list)
        requires['limit_down'] = limit_down != True

    if other_limit is not None:

        for key in other_limit:
            other = pd.read_hdf('%s/%s.h5' % (address, key), key).reindex_like(stock_list).rank(axis=1, pct=True)
            limit = other_limit[key] if isinstance(other_limit[key], list) else [other_limit[key]]
            requires[key] = True
            for i in limit:
                requires[key] &= other > i if i < 0.5 else other <= i

    if requires.__len__() > 1:
        stock_list = reduce(lambda x, y: x & y, requires.values())

    stock_list = stock_list.shift(1).iloc[1:] if trade_mode else stock_list

    date_list = stock_list.sum(axis=1) > 0.5
    start_date = date_list[date_list].index[0] if not start_date else max(date_list[date_list].index[0], start_date)
    stock_list = stock_list.loc[start_date : end_date]
    code_list = stock_list.sum() > 0.5
    code_list = sorted(code_list[code_list].index.to_list())
    stock_list = stock_list.reindex(columns=code_list) > 0.5

    return stock_list

