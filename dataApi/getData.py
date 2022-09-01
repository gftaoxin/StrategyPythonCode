import os
import pickle
import pandas as pd
import numpy as np
import datetime as dt
from xquant.marketdata import MarketData
from xquant.factordata import FactorData
from xquant.thirdpartydata.marketdata import MarketData as Market3Data
from dataApi.stockList import trans_windcode2int, trans_int2windcode
from dataApi.indName import sw_level1, citics_level1
from dataApi.tradeDate import get_trade_date_interval, trans_datetime2int, get_minute_index, get_recent_trade_date, \
    get_pre_trade_date, get_date_range, get_desample_minute_panel, trade_minutes

fd = FactorData()
m3d = Market3Data()

base_address = '/data/group/800442/800319/junkData/'

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

def get_minute_1min(datetime, minute_interval=1, factor_list=None, code_list=None, type='stock', base_date=20130101, diy_address=None):

    if minute_interval == 1:
        freq = 242
        address = base_address + 'minuteByFactor'
    else:
        freq = 240 // minute_interval
        address = base_address + 'minuteDesampleByFactor'

    if type =='bench':
        address += 'Bench'

    if diy_address != None:
        address = diy_address

    if factor_list == None:
        factor_list = [x[:-3] for x in os.listdir(address)]
        if minute_interval != 1:
            factor_list = [x for x in factor_list if x.split('_')[-1][:-1].isdigit()]
            factor_list = [x for x in factor_list if int(x.split('_')[-1][:-1]) == minute_interval]

    if (code_list != None) & (type == 'stock'):
        code_list = [trans_windcode2int(x) for x in code_list]

    if isinstance(datetime, str):
        if datetime.isdigit():
            datetime = int(datetime)

    date, time = divmod(datetime, 10000)
    row = get_trade_date_interval(date, base_date) * freq + get_minute_index(time, minute_interval)

    df = pd.concat([pd.read_hdf('%s/%s.h5' % (address, factor), factor, start=row, stop=row+1).iloc[0]
                    .rename(factor) for factor in factor_list], axis=1)
    if code_list != None:
        df = df.reindex(code_list)
    return df

def get_minute_1factor(factor, start_datetime=None, end_datetime=None, minute_interval=1, code_list=None,
                       type='stock', base_date=20130101, diy_address=None):

    if minute_interval == 1:
        freq = 242
        address = base_address + 'minuteByFactor'
    else:
        freq = 240 // minute_interval
        address = base_address + 'minuteDesampleByFactor'

    if type =='bench':
        address += 'Bench'

    if diy_address != None:
        address = diy_address

    if (code_list is not None) & (type == 'stock'):
        code_list = [trans_windcode2int(x) for x in code_list]

    if isinstance(start_datetime, str):
        if start_datetime.isdigit():
            start_datetime = int(start_datetime)

    if isinstance(end_datetime, str):
        if end_datetime.isdigit():
            end_datetime = int(end_datetime)

    if start_datetime == None:
        start_row = 0
    elif start_datetime / 1e8 < 1:
        start_datetime = get_date_range(start_datetime)[0]
        start_row = get_trade_date_interval(start_datetime, base_date) * freq
    else:
        date, time = divmod(start_datetime, 10000)
        start_row = get_trade_date_interval(date, base_date) * freq + get_minute_index(time, minute_interval)

    if end_datetime == None:
        stop_row = None
    elif end_datetime / 1e8 < 1:
        stop_row = (get_trade_date_interval(end_datetime, base_date) + 1) * freq
    else:
        date, time = divmod(end_datetime, 10000)
        stop_row = get_trade_date_interval(date, base_date) * freq + get_minute_index(time, minute_interval) + 1

    df = pd.read_hdf('%s/%s.h5' % (address, factor), factor, start=start_row, stop=stop_row)
    if code_list is not None:
        df = df.reindex(columns=code_list)
    return df

def get_minute_1stock(code, start_datetime=None, end_datetime=None, factor_list=None,
                      desample_minute_period=1, type='stock', diy_address=None):

    if type == 'stock':
        address = base_address + 'minuteByStock'
        code = str(trans_windcode2int(code))
    elif type == 'bench':
        address = base_address + 'minuteByStockBench'
    else:
        raise TypeError("type must be stock or bench")

    if diy_address != None:
        address = diy_address

    if isinstance(start_datetime, str):
        if start_datetime.isdigit():
            start_datetime = int(start_datetime)

    if isinstance(end_datetime, str):
        if end_datetime.isdigit():
            end_datetime = int(end_datetime)

    if start_datetime == None:
        pass
    elif start_datetime / 1e8 < 1:
        start_date, start_time = start_datetime, 925
    else:
        start_date, start_time = divmod(start_datetime, 10000)

    if end_datetime == None:
        stop_row = None
    elif end_datetime / 1e8 < 1:
        stop_date, stop_time = end_datetime, 1500
        stop_row = (get_trade_date_interval(end_datetime, base_date=20130101) + 1) * 242
    else:
        stop_date, stop_time = divmod(end_datetime, 10000)
        stop_row = get_trade_date_interval(stop_date) * 242 + get_minute_index(stop_time) + 1

    df = pd.read_hdf('%s/%s.h5' % (address, code), code, stop=stop_row)

    if factor_list != None:
        df = df.reindex(columns=['date','time'] + factor_list)

    if start_datetime != None:
        df = df[((df['date'] == start_date) & (df['time'] >= start_time)) | (df['date'] > start_date)]

    if end_datetime != None:
        df = df[(df['date'] < stop_date) | ((df['date'] == stop_date) & (df['time'] <= stop_time))]

    if desample_minute_period >=2:
        methods = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'amt':'sum', 'vol':'sum', 'deal':'sum'}
        factors = [x for x in df.columns if x not in ('date', 'time')]
        methods = {x: methods[x] for x in factors}
        df = get_desample_minute_panel(df, desample_minute_period, methods)
    else:
        df = df.set_index(['date', 'time'])

    return df

def _trans_order_detail(x):
    return [int(float(_x)) for _x in x.replace('[', '').replace(']', '').split('|') if _x != '']

def _get_stock_short_tick(code, start_date, end_date, start_time=90000, end_time=160000):

    md = MarketData()
    start_date = str(int(start_date))
    end_date = str(int(end_date))
    start_time = str(int(start_time)).zfill(6)
    end_time = str(int(end_time)).zfill(6)
    df = md.get_data_by_time_frame(
        table_type='Stock',
        security_id=code,
        start_time_str=start_date + ' ' + start_time + '000',
        end_time_str=end_date + ' ' + end_time + '000',
        trading_phase_code = ['1', '2', '3', '5', '6'],
        sort_by_receive_time=False,
    )
    factors = ['MDDate', 'MDTime', 'TradingPhaseCode', 'PreClosePx', 'NumTrades', 'TotalVolumeTrade',
                 'TotalValueTrade', 'LastPx', 'OpenPx', 'HighPx', 'LowPx', 'DiffPx1', 'DiffPx2', 'MaxPx',
                 'MinPx', 'TotalBidQty', 'TotalOfferQty', 'WeightedAvgBidPx', 'WeightedAvgOfferPx', 'SLYOne',
                 'Buy1Price', 'Buy1OrderQty', 'Buy1NumOrders', 'Buy1NoOrders', 'Buy1OrderDetail', 'Sell1Price',
                 'Sell1OrderQty', 'Sell1NumOrders', 'Sell1NoOrders', 'Sell1OrderDetail', 'Buy2Price', 'Buy2OrderQty',
                 'Buy2NumOrders', 'Sell2Price', 'Sell2OrderQty', 'Sell2NumOrders', 'Buy3Price', 'Buy3OrderQty',
                 'Buy3NumOrders', 'Sell3Price', 'Sell3OrderQty', 'Sell3NumOrders', 'Buy4Price', 'Buy4OrderQty',
                 'Buy4NumOrders', 'Sell4Price', 'Sell4OrderQty', 'Sell4NumOrders', 'Buy5Price', 'Buy5OrderQty',
                 'Buy5NumOrders', 'Sell5Price', 'Sell5OrderQty', 'Sell5NumOrders', 'Buy6Price', 'Buy6OrderQty',
                 'Buy6NumOrders', 'Sell6Price', 'Sell6OrderQty', 'Sell6NumOrders', 'Buy7Price', 'Buy7OrderQty',
                 'Buy7NumOrders', 'Sell7Price', 'Sell7OrderQty', 'Sell7NumOrders', 'Buy8Price', 'Buy8OrderQty',
                 'Buy8NumOrders', 'Sell8Price', 'Sell8OrderQty', 'Sell8NumOrders', 'Buy9Price', 'Buy9OrderQty',
                 'Buy9NumOrders', 'Sell9Price', 'Sell9OrderQty', 'Sell9NumOrders', 'Buy10Price', 'Buy10OrderQty',
                 'Buy10NumOrders', 'Sell10Price', 'Sell10OrderQty', 'Sell10NumOrders', 'ReceiveDateTime']
    if len(df) == 0:
        return pd.DataFrame(columns=factors)
    else:
        df = df[factors]
        df[['MDDate', 'MDTime', 'TradingPhaseCode']] = df[['MDDate', 'MDTime', 'TradingPhaseCode']].applymap(int)
        df['MDTime'] = df['MDTime'] // 1000
        df[['Buy1OrderDetail', 'Sell1OrderDetail']] = df[['Buy1OrderDetail', 'Sell1OrderDetail']].applymap(_trans_order_detail)
        return df

def _get_bench_short_tick(code, start_date, end_date, start_time=90000, end_time=160000):

    start_date = str(int(start_date))
    end_date = str(int(end_date))
    start_time = str(int(start_time)).zfill(6)
    end_time = str(int(end_time)).zfill(6)
    df = m3d.getMDSecurityTickDataFrame(
        htscSecurityID=code,
        startDateTime=start_date + start_time,
        endDateTime=end_date + end_time,
        QueryType=1,
    )
    return df

def get_tick_1stock(code, start_datetime=20150105090000, end_datetime=None, type='stock'):

    benchmarks = {'HS300':'000300.SH', 'ZZ500':'000905.SH', 'ZZ800':'000906.SH', 'SZZZ':'000001.SH', 'SZCZ':'399001.SZ',
                  'ZZ1000':'000852.SH', 'SZ50':'000016.SH', 'ZXBZ':'399101.SZ', 'CYBZ':'399102.SZ'}

    if code in benchmarks.keys():
        code = benchmarks[code]
        type = 'bench'
    elif code in benchmarks.values():
        type = 'bench'
    else:
        code = trans_int2windcode(code)

    if isinstance(start_datetime, str):
        start_datetime = int(start_datetime)

    if isinstance(end_datetime, str):
        end_datetime = int(end_datetime)

    if start_datetime / 100000000 < 1:
        start_datetime = start_datetime * 1000000 + 90000

    if end_datetime is None:
        end_datetime = get_recent_trade_date(dividing_point=15) * 1000000 + 160000

    if end_datetime / 100000000 < 1:
        end_datetime = end_datetime * 1000000 + 160000

    start_date, start_time = divmod(start_datetime, 1000000)
    end_date, end_time = divmod(end_datetime, 1000000)

    if type == 'stock':
        _start_date, _start_time = start_date, start_time
        _end_date = int(dt.datetime.strftime(dt.datetime.strptime(str(start_date), '%Y%m%d') + dt.timedelta(60), '%Y%m%d'))
        df = pd.DataFrame()
        while _end_date < end_date:
            temp_df = _get_stock_short_tick(code, _start_date, _end_date, _start_time)
            df = df.append(temp_df, ignore_index=True)
            _start_date = int(dt.datetime.strftime(dt.datetime.strptime(str(_end_date), '%Y%m%d')
                                                   + dt.timedelta(1), '%Y%m%d'))
            _end_date = int(dt.datetime.strftime(dt.datetime.strptime(str(_start_date), '%Y%m%d')
                                                   + dt.timedelta(60), '%Y%m%d'))
            _start_time = 90000
        temp_df = _get_stock_short_tick(code, _start_date, end_date, _start_time, end_time)
        df = df.append(temp_df, ignore_index=True)

    elif type == 'bench':
        if start_time == 90000:
            start_time = 0
        if end_time == 160000:
            end_time = 235959
        _start_date, _start_time = start_date, start_time
        df = pd.DataFrame()
        while _start_date < end_date:
            temp_df = _get_bench_short_tick(code, _start_date, _start_date, _start_time)
            df = df.append(temp_df)
            if code[-2:] not in ('SH', 'SZ'):
                _start_date = int(dt.datetime.strftime(dt.datetime.strptime(str(_start_date), '%Y%m%d')
                                                       + dt.timedelta(1), '%Y%m%d'))
            else:
                _start_date = get_pre_trade_date(_start_date, -1)
            _start_time = 0
        temp_df = _get_bench_short_tick(code, _start_date, end_date, _start_time, end_time)
        df = df.append(temp_df)

    else:
        raise ValueError("type must be stock or bench")

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

def get_minute_return(minute_interval, start_date=None, end_date=None, code_list=None, price='close', diy_address=None):

    if minute_interval == 1:
        factor = price
    else:
        factor = '%s_%sm' % (price, minute_interval)

    close = get_minute_1factor(factor, start_datetime=start_date, end_datetime=end_date,
                               minute_interval=minute_interval, code_list=code_list, diy_address=diy_address)

    if close.index[0][0] == 20130104:
        _start_date = 20130104
    else:
        _start_date = get_pre_trade_date(close.index[0][0])

    _end_date = close.index[-1][0]
    date_list = get_date_range(_start_date, _end_date)

    adjfactor = get_daily_1factor('adjfactor', date_list=date_list, code_list=code_list).reindex(columns=close.columns)
    dclose = get_daily_1factor('close', date_list=date_list, code_list=code_list).reindex(columns=close.columns)
    adjfactor = adjfactor / (adjfactor.shift(1) * dclose.shift(1))

    if close.index[0][0] != 20130104:
        adjfactor = adjfactor.iloc[1:]

    ret = close.pct_change()
    n = 240 // minute_interval if minute_interval >= 2 else 242
    ret.iloc[0::n] = (close.iloc[0::n].reset_index(drop=True) * adjfactor.reset_index(drop=True)
                       - 1).set_index(close.iloc[0::n].index)
    return ret

def get_minute_excess_return(minute_interval, start_date=None, end_date=None, bench='HS300',
                             code_list=None, diy_address=None, price='close'):

    ret = get_minute_return(minute_interval, start_date, end_date, code_list, price, diy_address)

    if minute_interval == 1:
        factor = price
    else:
        factor = '%s_%sm' % (price, minute_interval)

    if ret.index[0][0] == 20130104:
        _start_date_time = 20130104
    else:
        _start_date_time = get_pre_trade_date(ret.index[0][0]) * 10000 + 1500
    _end_date = ret.index[-1][0]

    ret_bench = get_minute_1factor(factor, _start_date_time, _end_date, minute_interval=minute_interval,
                                   code_list=[bench], type='bench').pct_change()

    if ret.index[0][0] != 20130104:
        ret_bench = ret_bench.iloc[1:, 0]
    else:
        ret_bench = ret_bench.iloc[:, 0]

    ret = ret.sub(ret_bench, axis=0)
    return ret

def get_modified_ind_mv(date_list=None, code_list=None, ind_type='SW'):

    mv = np.log(get_daily_1factor('mkt_cap_ard', date_list, code_list).values)
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

def get_minute_pickle(factor, date_list, code_list=None,
                      address='/data/group/800080/PanelMinDataForZT/stock/', type='stock'):

    if type == 'bench':
        address = address + '/../index/'

    start_date = date_list[0]
    end_date = date_list[-1]

    month_list = sorted(list(set(get_date_range(start_date, end_date, 'M') + [end_date])))
    short_month_list = sorted(list({x // 100 for x in month_list}))
    month_start = get_recent_trade_date(short_month_list[0] * 100)
    month_end = get_recent_trade_date(short_month_list[-1] * 100)

    start_keep = get_trade_date_interval(start_date, month_start) * 242
    end_keep = (get_trade_date_interval(end_date, month_end) + 1) * 242
    df_list = [pd.read_pickle('%s/%s/%s_%s.pkl' % (address, factor, x, factor)) for x in short_month_list]
    df_list[-1] = df_list[-1].iloc[:end_keep] if len(month_list) > 1 else df_list[-1].iloc[start_keep: end_keep]
    df_list[0] = df_list[0].iloc[start_keep:] if len(month_list) > 1 else df_list[0]

    df = pd.concat(df_list)
    df.columns = df.columns.map(trans_windcode2int)
    df = df.reindex(columns=code_list)
    df.index = pd.MultiIndex.from_product([date_list, trade_minutes])
    return df

if __name__ == '__main__':
    df = get_quarter_1factor('tot_oper_rev')
    df = get_ind_neutral(get_quarter_1factor('tot_oper_rev'))