# # 函数完成
import pandas as pd
import numpy as np
from dataApi.tradeDate import  get_recent_trade_date, trans_datetime2int, get_trade_date_interval
from functools import reduce
from BasicData.local_path import *

stock_address = base_address+'daily/'
bench_address = base_address+'dailyBench/'

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
def get_code_list(address = base_address+'daily/'):
    code_list = pd.read_pickle(address + 'code_list.pkl')
    return code_list

def get_stock_list(date=None,address = base_address+'daily/'):
    if date != None:
        date = trans_datetime2int(date)
    date = get_recent_trade_date(date)
    row = get_trade_date_interval(date)
    stock_list = pd.read_hdf(address + '/stock_list.h5', 'stock_list', start=row, stop=row + 1).iloc[0]
    stock_list = stock_list[stock_list].index.to_list()

    return stock_list

# 函数3：通常默认股票池尾非STPT，至少上市1年，没有停牌，复牌至少一天
def clean_stock_list(stock_list='ALL', no_ST=True, least_live_days=240, no_pause=True, least_recover_days=1,
                     no_pause_limit=0.5, no_pause_stats_days=120, no_limit_up=False, no_limit_down=False,
                     start_date=None, end_date=None, trade_mode=False,address=base_address):

    today = get_recent_trade_date(dividing_point=8.8 if trade_mode else 17.3)

    requires = {}
    if stock_list == 'ALL':
        stock_list = pd.read_hdf('%sdaily/stock_list.h5' % address , 'stock_list')
    elif stock_list in ('HS300', 'ZZ500', 'ZZ800', 'SZ50'):
        stock_list = pd.read_hdf('%sdaily/%s.h5' % (address, stock_list), stock_list)
        stock_list = stock_list>0
    else:
        raise ValueError("stock_list must in (ALL, HS300, ZZ500, ZZ800, SZ50).")

    stock_list = stock_list.loc[:today].shift(-1) > 0.5 if trade_mode else stock_list.loc[:today] > 0.5
    requires['stock_list'] = stock_list

    use_address = address + 'daily/'
    if no_ST:
        ST = pd.read_hdf('%s/ST.h5' % use_address, 'ST').reindex_like(stock_list)
        requires['ST'] = ST != True

    if least_live_days >= 2:
        live_days = pd.read_hdf('%s/stock_list.h5' % use_address, 'stock_list').cumsum()
        live_days = live_days.shift(-1) if trade_mode else live_days
        live_days = live_days.reindex_like(stock_list)
        requires['live_days'] = ((live_days >= least_live_days) | (live_days.T == live_days.max(axis=1)).T)

    if no_pause:
        pause = pd.read_hdf('%s/pause.h5' % use_address, 'pause')
        pause = pause.shift(-1) if trade_mode else pause
        pause1 = pause[pause > 0.5].ffill(limit=least_recover_days-1) if least_recover_days >= 2 else pause[pause == True]
        requires['pause'] = pause1.isnull().reindex_like(stock_list)

        if no_pause_limit > 0:
            pause2 = pause.rolling(no_pause_stats_days, min_periods=1).sum() / pause.rolling(no_pause_stats_days, min_periods=1).count()
            requires['pause2'] = pause2.isnull().reindex_like(stock_list) < 0.5

    if no_limit_up:
        limit_up = pd.read_hdf('%s/limit_up.h5' % address, 'limit_up').reindex_like(stock_list)
        requires['limit_up'] = limit_up != True

    if no_limit_down:
        limit_down = pd.read_hdf('%s/limit_down.h5' % address, 'limit_down').reindex_like(stock_list)
        requires['limit_down'] = limit_down != True

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


# 函数4：获取申万和中信行业代码对应的名称
def get_ind_con(ind_type='sw',level=1):
    # 输入：sw,sw2021,CITICS,获取申万和中信对应的指数代码，指数名称
    level_dict = {1: ['一级行业代码', '一级行业名称'],
                  2: ['二级行业代码', '二级行业名称'],
                  3: ['三级行业代码', '三级行业名称']}

    if ind_type == 'sw_all':
        ind_name1 = pd.read_excel(base_address + '行业分类.xlsx', sheet_name='SW2021')
        ind_name2 = pd.read_excel(base_address + '行业分类.xlsx', sheet_name='SW')
        level = int(level) if type(level) == str else level
        if type(level) == int:
            dict_data1 = ind_name1[level_dict[level]].set_index(level_dict[level][0])[level_dict[level][1]].to_dict()
            dict_data2 = ind_name2[level_dict[level]].set_index(level_dict[level][0])[level_dict[level][1]].to_dict()
            dict_data = dict(dict_data2, **dict_data1)

        elif type(level) == list:
            dict_data = dict()
            for l in level:
                dict_data1 = ind_name1[level_dict[l]].set_index(level_dict[l][0])[level_dict[l][1]].to_dict()
                dict_data2 = ind_name2[level_dict[l]].set_index(level_dict[l][0])[level_dict[l][1]].to_dict()
                dict_data = dict(dict_data, **dict_data1)
                dict_data = dict(dict_data, **dict_data2)
        else:
            raise ValueError("Use SW_new must use level as int or str ")
        return dict_data
    else:
        ind_name = pd.read_excel(base_address + '行业分类.xlsx',sheet_name=ind_type)
        level = int(level) if type(level) == str else level
        if type(level) == int:
            dict_data = ind_name[level_dict[level]].set_index(level_dict[level][0])[level_dict[level][1]].to_dict()
        elif type(level) == list:
            df = pd.Series()
            for l in level:
                df = pd.concat([df,ind_name[level_dict[l]].set_index(level_dict[l][0])[level_dict[l][1]].drop_duplicates()])
            dict_data = df.to_dict()
        else:
            raise ValueError("level type should be int or list ")
        return dict_data
