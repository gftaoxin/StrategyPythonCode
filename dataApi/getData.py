import pickle
import pandas as pd
import numpy as np
import cx_Oracle
from dataApi.stockList import trans_windcode2int, trans_int2windcode, get_code_list
from dataApi.indName import sw_level1, citics_level1
from dataApi.tradeDate import get_trade_date_interval, trans_datetime2int, get_recent_trade_date, \
    get_pre_trade_date, get_date_range
from dataApi.stockList import get_ind_con
from BasicData.local_path import *
con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True) # 写入信号

def load_pickle(file):

    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data
# 基于日频信息获取数据
def get_daily_1day(factor_list, date=None, code_list=None, type='stock', base_date=base_date):

    row = get_trade_date_interval(date, base_date)

    if type == 'stock':
        address = base_address + 'daily/'
        if code_list == None:
            code_list = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list', start=row, stop=row+1).iloc[0]
            code_list = code_list[code_list].index.to_list()
        else:
            code_list = [trans_windcode2int(x) for x in code_list]
    elif type == 'bench':
        address = base_address + 'dailyBench/'
        if code_list == None:
            code_list = ['HS300', 'ZZ500', 'ZZ800', 'SZZZ', 'SZCZ', 'ZZ1000', 'SZ50', 'ZXBZ', 'CYBZ','wind_A','avg_A']
    elif type == 'SW':
        address = base_address + 'dailyBench/'
        factor_list = ['sw_'+x if x[:2]!='sw' else x for x in factor_list]
        if code_list == None:
            code_list = get_ind_con('SW2021',[1,2,3]).keys()
    elif type == 'CITICS':
        address = base_address + 'dailyBench/'
        factor_list = ['zx_' + x if x[:2] != 'zx' else x for x in factor_list]
        if code_list == None:
            code_list = get_ind_con('CITICS', [1, 2, 3]).keys()
    else:
        raise TypeError("type must be stock or bench or ind")

    df = pd.concat([pd.read_hdf('%s/%s.h5' % (address, factor), factor, start=row, stop=row+1).iloc[0]
                    .rename(factor) for factor in factor_list], axis=1)
    if code_list != None:
        df = df.reindex(code_list)
    return df

def get_daily_1stock(code, factor_list, date_list=None, type='stock'):

    if type == 'stock':
        address = base_address + 'daily'
        code = trans_windcode2int(code)
    elif type == 'bench':
        address = base_address + 'dailyBench'
    else:
        raise TypeError("type must be stock or bench")

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

def get_daily_1factor(factor, date_list=None, code_list=None, type='stock'):

    if type == 'stock':
        address = base_address + 'daily'
        if code_list != None:
            code_list = [trans_windcode2int(x) for x in code_list]
    elif type == 'bench':
        address = base_address + 'dailyBench'
    else:
        raise TypeError("type must be stock or bench")

    df = pd.read_hdf('%s/%s.h5' % (address, factor), factor)
    if date_list != None:
        _date_list = [trans_datetime2int(x) for x in date_list]
        df = df.reindex(index=_date_list)
    if code_list != None:
        df = df.reindex(columns=code_list)
    return df

# 基于财报信息获取数据，这部分不做存储，直接从数据库中导入
def _trans_financial2fixed_date(date):
    # 金融数据日期对齐
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

# 该调用方法只适合有报告期的财务数据：符合条件的有：且该数据对应的是报告期（和使用周期存在差异）
# 中国A股资产负债表[AShareBalanceSheet]，中国A股利润表[AShareIncome]，中国A股现金流量表[AShareCashFlow]，中国A股业绩快报[AShareProfitExpress]，
def get_quarter_1factor(factor,table, report_type = '408001000',code_list=None, date_list=None):
    #报表类型:    408001000:    合并报表           408004000:    合并报表(调整)     408005000:    合并报表(更正前)
    #            408050000:    合并调整(更正前)    408006000:    母公司报表        408009000:    母公司报表(调整)
    #            408010000:    母公司报表(更正前)  408060000:    母公司调整(更正前)
    report_type_list = ['AShareBalanceSheet','AShareIncome','AShareCashFlow']
    no_type_list = ['AShareProfitExpress']
    if code_list != None:
        code_list = [trans_int2windcode(x) for x in code_list]
    else:
        code_list = [trans_int2windcode(x) for x in get_code_list()]
    if date_list != None:
        date_list = get_date_range(20090331, None, 'R')

    begin,end = int(date_list[0]), int(date_list[-1])

    factor_str = factor + ',REPORT_PERIOD,S_INFO_WINDCODE,STATEMENT_TYPE'
    if table in report_type_list:
        sql = r"select %s from wind.%s a where a.REPORT_PERIOD >= '%s' and a.STATEMENT_TYPE = '%s'" % (factor_str, table,str(begin),report_type)
    else:
        sql = r"select %s from wind.%s a where a.REPORT_PERIOD >= '%s'" % (factor_str, table,str(begin))
    data_values = pd.read_sql(sql, con)

    data_values = data_values.pivot_table(index='REPORT_PERIOD',columns='S_INFO_WINDCODE',values=factor)
    data_values.index = data_values.index.map(int)
    data_values = data_values.reindex(columns=code_list).loc[begin:end]
    data_values.columns = data_values.columns.map(trans_windcode2int)

    return data_values.dropna(how='all')

# 获取单个季度的财财报指标，计算ttm,yoy,gog等指标（主要为利润表和现金流量表）
def get_single_quarter(factor, table, report_type = '408002000',code_list=None, date_list=None):

    if code_list != None:
        code_list = [trans_int2windcode(x) for x in code_list]

    df = get_quarter_1factor(factor,table, report_type, code_list=code_list)
    #df_sig = df.diff()
    #df_sig[::4] = df[::4]  如果获取的是最新的数据，则需要把每一年的第一个季度结果保留，不做diff
    if date_list != None:
        df = df.reindex(date_list)
    return df

def get_ttm_quarter(factor, table, report_type='408002000', code_list=None, date_list=None):

    if code_list != None:
        code_list = [trans_int2windcode(x) for x in code_list]

    df = get_single_quarter(factor, table, report_type, code_list=code_list)
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

# 使用固定日期将季度数据填充为日度数据：同时由于年报和第二年的一季度报同为4月30日更新，此时选择first为保留年报，选择last为一季报
def fill_quarter2daily_by_fixed_date(df, keep = 'last'):

    _df = df.copy().sort_index()
    _df.index = _df.index.map(_trans_financial2fixed_date)
    _df = _df[~_df.index.duplicated(keep=keep)].replace(np.nan, np.inf)
    _df = _df.reindex(get_date_range(_df.index[0])).ffill().replace(np.inf, np.nan)
    return _df

def fill_quarter2daily_by_issue_date(df, table, report_type, keep = 'last'):

    _df = df.copy()
    _df.index.name = 'mddate'
    _df.columns.name = 'stock'

    date_list = _df.index.map(trans_datetime2int).map(str).to_list()
    code_list = _df.columns.map(trans_int2windcode).to_list()
    # 获取披露日期
    issue_date = get_quarter_1factor('ANN_DT',table, report_type, code_list=code_list, date_list=date_list)
    issue_date.index.name = 'mddate'
    issue_date.columns.name = 'stock'
    issue_date = issue_date.stack().rename('date')

    _df = pd.concat([_df.stack(), issue_date], axis=1).reset_index()
    _df.loc[_df['date'].isnull(), 'date'] = _df.loc[_df['date'].isnull(), 'mddate'].map(_trans_financial2fixed_date) # 如果当期没有预告日期，则变为财报末尾日期
    _df = _df.rename(columns={'stock': 'code'})
    _df = _df.sort_values(['code', 'mddate']).drop_duplicates(subset=['code', 'date'], keep='last')
    # 为了处理年报和一季报的问题，进行调整
    if keep == 'first': # 保留年报
        _df = _df[(_df['mddate']%10000 != 331)].replace(np.inf,np.nan)
    if keep == 'last':
        _df = _df[(_df['mddate'] % 10000 != 1231)].replace(np.inf,np.nan)

    _df = _df.pivot('date', 'code', 0)

    _df = _df.reindex(get_date_range(_df.index[0])).ffill().replace(np.inf,np.nan)
    return _df


# 其余数据中性化处理（尚未处理）
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