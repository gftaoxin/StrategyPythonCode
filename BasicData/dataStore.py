import os,datetime
import pickle
import time
import pandas as pd
import numpy as np
import cx_Oracle
import gc,re
from BasicData.local_path import *

con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True) # 写入信号

# 最开始(运行第一步）：存储股票代码，和存储交易日历，通常1个月更新一次即可，无需重复重新
def get_list_factor(save_path = base_address+'daily/'):
    # 确认路径是否存在，不存在创建路径
    os.makedirs(save_path) if os.path.exists(save_path) == False else print('存储路径已存在')
    # 1、存储交易日历
    trade_table,trade_str = 'AShareCalendar', 'TRADE_DAYS'
    sql = r"select %s from wind.%s" % (trade_str, trade_table)
    data_values = pd.read_sql(sql, con).sort_values(by=trade_str)
    trade_date = data_values[trade_str][data_values[trade_str]>='20090101'].astype(int).drop_duplicates().to_list()
    # 交易日周期的dataframe：每个周最后一个交易日，每个月最后一个交易日，每个季度最后一个交易日，每半年最后一个交易日，每年最后一个交易日
    trade_calendar_df = pd.DataFrame(0,index=trade_date,columns=['WeekEnd','MonthEnd','QuarterEnd','HalfYearEnd','YearEnd'])

    previous_i = -1
    previous_open_week,previous_open_month = -1, -1
    previous_open_halfyear,previous_open_year = -1, -1

    for i in trade_calendar_df.index:
        date = datetime.datetime.strptime(str(i), '%Y%m%d').date()
        # 设置WeekEnd
        current_open_week = date.isocalendar()[1]
        if (current_open_week != previous_open_week) & (previous_open_week != -1):
            trade_calendar_df.loc[previous_i, 'WeekEnd'] = 1
        # 设置MonthEnd
        current_open_month = date.month
        if (current_open_month != previous_open_month) & (previous_open_month != -1):
            trade_calendar_df.loc[previous_i, 'MonthEnd'] = 1
            # 根据月份设置QuarterEnd
            if (current_open_month in [1, 4, 7, 10]) & (previous_open_month != -1):
                trade_calendar_df.loc[previous_i, 'QuarterEnd'] = 1
            # 根据月份设置HalfYearEnd：
            if (current_open_month in [1, 7]) & (previous_open_month != -1):
                trade_calendar_df.loc[previous_i, 'HalfYearEnd'] = 1
       # 设置YearEnd
        current_open_year = date.year
        if (current_open_year != previous_open_year) & (previous_open_year != -1):
            trade_calendar_df.loc[previous_i, 'YearEnd'] = 1

        previous_i = i
        previous_open_week = current_open_week
        previous_open_month = current_open_month
        previous_open_year = current_open_year

    trade_calendar_df.to_pickle(save_path+'trade_calendar.pkl')
    with open(save_path +'trade_date.pkl','wb') as f:
        pickle.dump(trade_date,f)  # 保存交易日历

    # 2、存储股票代码，即每日可交易个股
    code_table = 'AShareDescription'
    code_str = 'S_INFO_WINDCODE,S_INFO_CODE,S_INFO_NAME,S_INFO_LISTDATE,S_INFO_DELISTDATE'
    sql = r"select %s from wind.%s" % (code_str, code_table)
    code_df = pd.read_sql(sql, con)
    wind_code = code_df['S_INFO_WINDCODE'].apply(lambda x:np.nan if ('A' in x or 'T' in x or 'BJ' in x) else x).dropna()
    code_df = code_df[code_df['S_INFO_WINDCODE'].isin(wind_code)].sort_values(by='S_INFO_WINDCODE')
    code_df = code_df[(code_df['S_INFO_DELISTDATE'].isna() == True) | (code_df['S_INFO_DELISTDATE']>='20100101')]

    code_df['S_INFO_CODE'] = code_df['S_INFO_CODE'].astype(int)
    code_list = code_df['S_INFO_CODE'].to_list()  # 股票列表
    code_df.set_index('S_INFO_CODE',inplace = True)

    code_df.to_pickle(save_path + 'code_name.pkl')
    with open(save_path + 'code_list.pkl', 'wb') as f:
        pickle.dump(code_list, f)  # 保存交易日历

# 1、储存个股基础数据：把所有字符串日期，改为数字日期，初始日期变成特定日期20100101
save_stockdaily_dict = {
    # 个股日行情：高开低手，复权价，福全因子，均价，涨跌停价格
    'AShareEODPrices': {
        'open': 'S_DQ_OPEN', 'high': 'S_DQ_HIGH', 'low': 'S_DQ_LOW', 'close': 'S_DQ_CLOSE','pre_close': 'S_DQ_PRECLOSE',
        'pct_chg': 'S_DQ_PCTCHANGE', 'volume': 'S_DQ_VOLUME', 'amt': 'S_DQ_AMOUNT',
        'open_badj': 'S_DQ_ADJOPEN', 'high_badj': 'S_DQ_ADJHIGH','low_badj': 'S_DQ_ADJLOW','close_badj': 'S_DQ_ADJCLOSE',
        'pre_close_badj':'S_DQ_ADJPRECLOSE','adjfactor': 'S_DQ_ADJFACTOR', 'vwap': 'S_DQ_AVGPRICE',
        'limit_up': 'S_DQ_LIMIT', 'limit_down': 'S_DQ_STOPPING'},
    # 个股每日数据：估值，换手率，市值，股本
    'AShareEODDerivativeIndicator':{
        'mkt_cap_ard':'S_VAL_MV', 'mkt_free_cap':'S_DQ_MV',
        'total_shares':'TOT_SHR_TODAY', 'A_shares':'FLOAT_A_SHR_TODAY', 'float_A_shares':'FREE_SHARES_TODAY', 'free_float_shares':'FREE_SHARES_TODAY',
        'turn':'S_DQ_TURN','free_turn':'S_DQ_FREETURNOVER',
        'pb':'S_VAL_PB_NEW',
        'pe':'S_VAL_PE', 'pe_ttm':'S_VAL_PE_TTM',
        'pcf_ocf':'S_VAL_PCF_OCF','pcf_ocf_ttm':'S_VAL_PCF_OCFTTM',
        'pcf_ncf':'S_VAL_PCF_NCF','pcf_ncf_ttm':'S_VAL_PCF_NCFTTM',
        'ps':'S_VAL_PS', 'ps_ttm':'S_VAL_PS_TTM',
        'limit_state':'UP_DOWN_LIMIT_STATUS',
        
        'net_profit_parent_comp_ttm':'NET_PROFIT_PARENT_COMP_TTM','net_profit_parent_comp_lyr':'NET_PROFIT_PARENT_COMP_LYR',
        'net_assets_today':'NET_ASSETS_TODAY',
        'net_cash_flows_oper_act_ttm':'NET_CASH_FLOWS_OPER_ACT_TTM', 'net_cash_flows_oper_act_lyr':'NET_CASH_FLOWS_OPER_ACT_LYR',
        'oper_rev_ttm':'OPER_REV_TTM', 'oper_rev_lyr':'OPER_REV_LYR',
        'net_incr_cash_cash_equ_ttm':'NET_INCR_CASH_CASH_EQU_TTM',
        'net_incr_cash_cash_equ_lyr':'NET_INCR_CASH_CASH_EQU_LYR',
    },
    # 个股日收益率
    'AShareYield':{
        'beta_day_1y':'BETA_DAY_1Y','beta_100w':'BETA_100W','beta_24m':'BETA_24M','beta_60m':'BETA_60M',
        'alpha_day_1y':'ALPHA_DAY_1Y','alpha_100w':'ALPHA_100W','alpha_24m':'ALPHA_24M','alpha_60m':'ALPHA_60M',
    },
    # 个股资金流向数据
    'AShareMoneyFlow':{
        'deal_num':'TRADES_COUNT',
        'BUY_VALUE_EXLARGE_ORDER':'BUY_VALUE_EXLARGE_ORDER','SELL_VALUE_EXLARGE_ORDER':'SELL_VALUE_EXLARGE_ORDER',
        'BUY_VALUE_LARGE_ORDER':'BUY_VALUE_LARGE_ORDER','SELL_VALUE_LARGE_ORDER':'SELL_VALUE_LARGE_ORDER',
        'BUY_VALUE_MED_ORDER':'BUY_VALUE_MED_ORDER','SELL_VALUE_MED_ORDER':'SELL_VALUE_MED_ORDER',
        'BUY_VALUE_SMALL_ORDER':'SELL_VALUE_SMALL_ORDER',
        'BUY_VOLUME_EXLARGE_ORDER':'BUY_VOLUME_EXLARGE_ORDER','SELL_VOLUME_EXLARGE_ORDER':'SELL_VOLUME_EXLARGE_ORDER',
        'BUY_VOLUME_LARGE_ORDER':'BUY_VOLUME_LARGE_ORDER','SELL_VOLUME_LARGE_ORDER':'SELL_VOLUME_LARGE_ORDER',
        'BUY_VOLUME_MED_ORDER':'BUY_VOLUME_MED_ORDER','SELL_VOLUME_MED_ORDER':'SELL_VOLUME_MED_ORDER',
        'BUY_VOLUME_SMALL_ORDER':'BUY_VOLUME_SMALL_ORDER','SELL_VOLUME_SMALL_ORDER':'SELL_VOLUME_SMALL_ORDER',
        'BUY_TRADES_EXLARGE_ORDER':'BUY_TRADES_EXLARGE_ORDER','SELL_TRADES_EXLARGE_ORDER':'SELL_TRADES_EXLARGE_ORDER',
        'BUY_TRADES_LARGE_ORDER':'BUY_TRADES_LARGE_ORDER','SELL_TRADES_LARGE_ORDER':'SELL_TRADES_LARGE_ORDER',
        'BUY_TRADES_MED_ORDER':'BUY_TRADES_MED_ORDER','SELL_TRADES_MED_ORDER':'SELL_TRADES_MED_ORDER',
        'BUY_TRADES_SMALL_ORDER':'BUY_TRADES_SMALL_ORDER','SELL_TRADES_SMALL_ORDER':'SELL_TRADES_SMALL_ORDER',
        'S_MFD_INFLOWVOLUME':'S_MFD_INFLOWVOLUME','S_MFD_INFLOW':'S_MFD_INFLOW',
        'S_MFD_INFLOW_OPENVOLUME':'S_MFD_INFLOW_OPENVOLUME','S_MFD_INFLOW_CLOSEVOLUME':'S_MFD_INFLOW_CLOSEVOLUME',
        'S_MFD_INFLOW_OPEN':'S_MFD_INFLOW_OPEN','S_MFD_INFLOW_CLOSE':'S_MFD_INFLOW_CLOSE',
        'MONEYFLOW_PCT_VOLUME':'MONEYFLOW_PCT_VOLUME',
        'OPEN_MONEYFLOW_PCT_VOLUME':'OPEN_MONEYFLOW_PCT_VOLUME','CLOSE_MONEYFLOW_PCT_VOLUME':'CLOSE_MONEYFLOW_PCT_VOLUME',
        'MONEYFLOW_PCT_VALUE':'MONEYFLOW_PCT_VALUE',
        'OPEN_MONEYFLOW_PCT_VALUE':'OPEN_MONEYFLOW_PCT_VALUE','CLOSE_MONEYFLOW_PCT_VALUE':'CLOSE_MONEYFLOW_PCT_VALUE',
        'S_MFD_INFLOWVOLUME_LARGE_ORDER':'S_MFD_INFLOWVOLUME_LARGE_ORDER',
        'NET_INFLOW_RATE_VOLUME_L':'NET_INFLOW_RATE_VOLUME_L','S_MFD_INFLOW_LARGE_ORDER':'S_MFD_INFLOW_LARGE_ORDER',
        'NET_INFLOW_RATE_VALUE_L':'NET_INFLOW_RATE_VALUE_L','MONEYFLOW_PCT_VOLUME_L':'MONEYFLOW_PCT_VOLUME_L',
        'MONEYFLOW_PCT_VALUE_L':'MONEYFLOW_PCT_VALUE_L','S_MFD_INFLOW_OPENVOLUME_L':'S_MFD_INFLOW_OPENVOLUME_L',
        'OPEN_NET_INFLOW_RATE_VOLUME_L':'OPEN_NET_INFLOW_RATE_VOLUME_L','S_MFD_INFLOW_OPEN_LARGE_ORDER':'S_MFD_INFLOW_OPEN_LARGE_ORDER',
        'S_MFD_INFLOW_CLOSEVOLUME_L':'S_MFD_INFLOW_CLOSEVOLUME_L','CLOSE_NET_INFLOW_RATE_VOLUME_L':'CLOSE_NET_INFLOW_RATE_VOLUME_L',
        'S_MFD_INFLOW_CLOSE_LARGE_ORDER':'S_MFD_INFLOW_CLOSE_LARGE_ORDER','CLOSE_NET_INFLOW_RATE_VALU_L':'CLOSE_MONEYFLOW_PCT_VOLUME_L',
        'BUY_VALUE_EXLARGE_ORDER_ACT':'BUY_VALUE_EXLARGE_ORDER_ACT','SELL_VALUE_EXLARGE_ORDER_ACT':'SELL_VALUE_EXLARGE_ORDER_ACT',
        'BUY_VALUE_LARGE_ORDER_ACT':'BUY_VALUE_LARGE_ORDER_ACT','SELL_VALUE_LARGE_ORDER_ACT':'SELL_VALUE_LARGE_ORDER_ACT',
        'BUY_VALUE_MED_ORDER_ACT':'BUY_VALUE_MED_ORDER_ACT','SELL_VALUE_MED_ORDER_ACT':'SELL_VALUE_MED_ORDER_ACT',
        'BUY_VALUE_SMALL_ORDER_ACT':'BUY_VALUE_SMALL_ORDER_ACT','SELL_VALUE_SMALL_ORDER_ACT':'SELL_VALUE_SMALL_ORDER_ACT',
    }
    }

def get_stock_factor(save_data_dict, start = base_date, resave = False):
    save_path = base_address + 'daily/'
    # resave = True：进行历史全周期数据的重刷使用  False：添加到该结果的后面
    end_date = str(get_recent_trade_date()) # 1、获取数据都区范围

    code_list = pd.read_pickle(save_path + 'code_list.pkl')
    code_list = [trans_int2windcode(x) for x in code_list]
    # 2、确认路径是否存在，不存在创建路径
    os.makedirs(save_path) if os.path.exists(save_path) == False else None
    # 3、针对每一个数据，进行存储，存储格式为单因子格式，date,code,factor
    for table in save_data_dict.keys():
        for data_name in save_data_dict[table].keys():
            old_name = save_data_dict[table][data_name]
            data_str = old_name +', TRADE_DT, S_INFO_WINDCODE'
            # （1）确定数据的提取日期，即start_date；如果不进行数据冲刷，且存储路径该结果存在，则调整start_date
            if (resave == False) & (os.path.exists(save_path + data_name + '.h5') == True):
                old_data = pd.read_hdf(save_path + data_name + '.h5',data_name)
                start_date = str(old_data.index[-1])
            else:
                old_data = pd.DataFrame()
                start_date = start
            # （2）获取信息
            if int(start_date) < int(end_date):
                sql = r"select %s from wind.%s a where a.TRADE_DT >= '%s' and  a.TRADE_DT <= '%s' " % \
                      (data_str,table,start_date, end_date)
                data_values = pd.read_sql(sql, con)

                data_values = data_values[data_values['S_INFO_WINDCODE'].isin(code_list)]
                save_data = data_values.pivot_table(values=old_name, index='TRADE_DT', columns='S_INFO_WINDCODE')
                # 转换数据格式
                save_data.index = save_data.index.astype(int)
                save_data.columns = pd.Series(save_data.columns).apply(lambda x:trans_windcode2int(x))

                save_data = pd.concat([old_data,save_data])
                save_data = save_data[~save_data.index.duplicated('last')]

                if save_data.index[0] > get_pre_trade_date(base_date, -1):
                    print(data_name+'数据存在问题，需要检查！！！！！！！！！！！！！！！！！！！！！！！')
                # 将该输出到保存地址中
                save_data = save_data.reindex(get_date_range(base_date,int(end_date)))
                save_data.to_hdf('%s/%s.h5' % (save_path,data_name), data_name, format='t')
                print(data_name + '存储完毕')
            else:
                print(data_name + '今日数据已更新')
            gc.collect()
# (1)获取指数成分股信息
def get_bench_weight(start = base_date,resave = False):
    save_path = base_address + 'daily/'
    icode_dict = {'SZ50': '000016.SH', 'HS300': '000300.SH', 'ZZ500': '000905.SH', 'ZZ800':'000906.SH'}
    icode_table = {'SZ50': 'AIndexSSE50Weight', 'HS300': 'AIndexHS300Weight',
                   'ZZ500': 'AIndexCSI500Weight', 'ZZ800':'AIndexCSI800Weight'}
    end_date = str(get_recent_trade_date()) # 1、获取数据都区范围
    os.makedirs(save_path) if os.path.exists(save_path) == False else None
    for index_name in icode_table.keys():
        index_table = icode_table[index_name]
        index_value = ['TRADE_DT', 'S_CON_WINDCODE', 'I_WEIGHT'] if index_name == 'HS300' else \
            ['TRADE_DT', 'S_CON_WINDCODE', 'WEIGHT']
        index_value_str = re.sub('[\'\[\]]', '', str(index_value))
        if (resave == False) & (os.path.exists(save_path + 'weighted_'+index_name+'.h5') == True):
            old_data = pd.read_hdf(save_path + 'weighted_'+index_name+'.h5','weighted_'+index_name)
            start_date = str(old_data.index[-1])
        else:
            old_data = pd.DataFrame()
            start_date = start

        if int(start_date) < int(end_date):
            sql = r"select %s from wind.%s a where a.S_INFO_WINDCODE = '%s' and" \
                  r" a.TRADE_DT >= '%s' and  a.TRADE_DT <= '%s' " % \
                  (index_value_str, index_table, icode_dict[index_name], start_date, end_date)
            save_data = pd.read_sql(sql, con)
            if index_name == 'HS300':
                save_data = save_data.pivot_table('I_WEIGHT',index='TRADE_DT',columns='S_CON_WINDCODE')
            else:
                save_data = save_data.pivot_table('WEIGHT', index='TRADE_DT', columns='S_CON_WINDCODE')
            # 转换数据格式
            save_data.index = save_data.index.astype(int)
            save_data.columns = pd.Series(save_data.columns).apply(lambda x: trans_windcode2int(x))

            save_data = pd.concat([old_data, save_data])
            save_data = save_data[~save_data.index.duplicated('last')]

            if save_data.index[0] > max(get_pre_trade_date(base_date, -1),20120104):
                print(index_name + '权重数据存在问题，需要检查！！！！！！！！！！！！！！！！！！！！！！！')
            # 将该输出到保存地址中
            save_data = save_data.reindex(get_date_range(base_date, int(end_date)))
            save_data.to_hdf('%s/%s.h5' % (save_path, index_name), index_name, format='t')
# (2)获取行业成分股信息
def get_ind_weight(start= base_date):
    save_path = base_address + 'daily/'
    date_list, code_list = get_date_range(start), get_code_list()
    # 分别获取申万一级行业，申万二级行业，申万三级行业的个股归属情况
    SW_table = 'SWIndexMembers'
    SW_value = ['S_INFO_WINDCODE','S_CON_WINDCODE','S_CON_INDATE','S_CON_OUTDATE']
    SW_data_str = re.sub('[\'\[\]]', '', str(SW_value))

    sql = r"select %s from wind.%s " % (SW_data_str, SW_table)
    SW_data = pd.read_sql(sql, con)
    SW_data = SW_data[(SW_data['S_CON_OUTDATE'].isna()==True) | (SW_data['S_CON_OUTDATE']>=str(base_date))]

    SW_data['S_CON_WINDCODE'] = SW_data['S_CON_WINDCODE'].apply(lambda x:trans_windcode2int(x))
    SW_data['S_CON_OUTDATE'] = SW_data['S_CON_OUTDATE'].apply(pd.to_numeric,errors='ignore')
    SW_data['S_CON_INDATE'] = SW_data['S_CON_INDATE'].apply(pd.to_numeric,errors='ignore')

    for sw_type in ['SW2021','SW']:
        for level in (1,2,3):
            ind_name = get_ind_con(sw_type,level).keys()
            SW_level = SW_data[SW_data['S_INFO_WINDCODE'].isin(ind_name)]
            SW_code = pd.DataFrame(index=date_list,columns=code_list)
            for i in SW_level.index:
                code = SW_level.loc[i,'S_CON_WINDCODE']
                ind_start_date = SW_level.loc[i, 'S_CON_INDATE']
                ind_end_date = SW_level.loc[i, 'S_CON_OUTDATE']
                ind = SW_level.loc[i,'S_INFO_WINDCODE']

                SW_code.loc[ind_start_date:ind_end_date,code] = ind

            SW_code.to_hdf('%s/%s%s.h5' % (save_path, sw_type,str(level)), sw_type+str(level), format='t')

    # 分别获取中信一级行业，中信二级行业的个股归属情况
    for level in [1,2]:
        CITICS_table = 'AIndexMembersCITICS' if level==1 else 'AIndexMembersCITICS2'
        CITICS_value = ['S_INFO_WINDCODE', 'S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE']
        CITICS_data_str = re.sub('[\'\[\]]', '', str(CITICS_value))

        sql = r"select %s from wind.%s " % (CITICS_data_str, CITICS_table)
        CITICS_data = pd.read_sql(sql, con)
        CITICS_data = CITICS_data[(CITICS_data['S_CON_OUTDATE'].isna() == True) | (CITICS_data['S_CON_OUTDATE'] >= str(base_date))]

        CITICS_data['S_CON_WINDCODE'] = CITICS_data['S_CON_WINDCODE'].apply(lambda x: trans_windcode2int(x))
        CITICS_data['S_CON_OUTDATE'] = CITICS_data['S_CON_OUTDATE'].apply(pd.to_numeric, errors='ignore')
        CITICS_data['S_CON_INDATE'] = CITICS_data['S_CON_INDATE'].apply(pd.to_numeric, errors='ignore')

        ind_name = get_ind_con('CITICS', level).keys()
        CITICS_level = CITICS_data[CITICS_data['S_INFO_WINDCODE'].isin(ind_name)]
        CITICS_code = pd.DataFrame(index=date_list, columns=code_list)
        for i in CITICS_level.index:
            code = CITICS_level.loc[i, 'S_CON_WINDCODE']
            ind_start_date = CITICS_level.loc[i, 'S_CON_INDATE']
            ind_end_date = CITICS_level.loc[i, 'S_CON_OUTDATE']
            ind = CITICS_level.loc[i, 'S_INFO_WINDCODE']

            CITICS_code.loc[ind_start_date:ind_end_date, code] = ind

        CITICS_code.to_hdf('%s/CITICS%s.h5' % (save_path, str(level)), 'CITICS' + str(level), format='t')

# 2、储存指数基础数据：把所有字符串日期，改为数字日期，初始日期变成特定日期20100101
save_benchdaily_dict = {
    'AIndexEODPrices': {'open': 'S_DQ_OPEN', 'high': 'S_DQ_HIGH', 'low': 'S_DQ_LOW','close': 'S_DQ_CLOSE',
                        'pre_close': 'S_DQ_PRECLOSE', 'pct_chg': 'S_DQ_PCTCHANGE',
                        'volume': 'S_DQ_VOLUME', 'amt': 'S_DQ_AMOUNT'},
    'AIndexValuation':{
        'pe':'PE_LYR','pe_ttm':'PE_TTM',
        'pb':'PB_LF',
        'pcf':'PCF_LYR', 'pcf_ttm':'PCF_TTM',
        'ps':'PS_LYR', 'ps_ttm':'PS_TTM',
        'turn':'TURNOVER','free_turn':'TURNOVER_FREE',
        'dividend_yield':'DIVIDEND_YIELD',
        'peg_his':'PEG_HIS'}
}

def get_bench_factor(save_data_dict, start = base_date, resave = False):
    save_path = base_address + 'dailyBench/'
    # resave = True：进行历史全周期数据的重刷使用  False：添加到该结果的后面
    end_date = str(get_recent_trade_date())  # 1、获取数据都区范围
    index_dict = {'000001.SH': 'SZZZ', '399001.SZ': 'SZCZ', '399102.SZ': 'CYBZ', '399101.SZ': 'ZXBZ',
                  '000016.SH': 'SZ50', '000300.SH': 'HS300', '000905.SH': 'ZZ500',
                  '000906.SH': 'ZZ800', '000852.SH': 'ZZ1000',
                  '881001.WI': 'wind_A', '8841388.WI': 'avg_A', }
    # 2、确认路径是否存在，不存在创建路径
    os.makedirs(save_path) if os.path.exists(save_path) == False else None
    # 3、针对每一个数据，进行存储，存储格式为单因子格式，date,code,factor
    index_list = tuple(list(index_dict.keys()))
    for table in save_data_dict.keys():
        for data_name in save_data_dict[table].keys():
            old_name = save_data_dict[table][data_name]
            data_str = old_name + ', TRADE_DT, S_INFO_WINDCODE'
            # （1）确定数据的提取日期，即start_date；如果不进行数据冲刷，且存储路径该结果存在，则调整start_date
            if (resave == False) & (os.path.exists(save_path + data_name + '.h5') == True):
                old_data = pd.read_hdf(save_path + data_name + '.h5',data_name)
                start_date = str(old_data.index[-1])
            else:
                old_data = pd.DataFrame()
                start_date = start
            # （2）获取信息
            if int(start_date) < int(end_date):
                if table == 'AIndexEODPrices':
                    # 获取数据组1：wind全A数据'881001.WI'，全A等权指数'8841388.WI'
                    A_table = 'AIndexWindIndustriesEOD'
                    A_list = ('881001.WI','8841388.WI')
                    sql = r"select %s from wind.%s a where a.TRADE_DT >= '%s' and  a.TRADE_DT <= '%s' " \
                          r"and a.S_INFO_WINDCODE in %s" % \
                          (data_str, A_table, start_date, end_date, A_list)
                    A_values = pd.read_sql(sql, con)
                    A_values = A_values.pivot_table(values=old_name, index='TRADE_DT', columns='S_INFO_WINDCODE')
                else:
                    A_values = pd.DataFrame()
                # 获取数据组2：常规指数数据
                sql = r"select %s from wind.%s a where a.TRADE_DT >= '%s' and  a.TRADE_DT <= '%s' " \
                      r"and a.S_INFO_WINDCODE in %s" % \
                      (data_str, table, start_date, end_date, index_list)
                data_values = pd.read_sql(sql, con)
                save_data = data_values.pivot_table(values=old_name, index='TRADE_DT', columns='S_INFO_WINDCODE')

                save_data = pd.concat([save_data,A_values],axis=1)
                # 转换数据格式
                save_data.index = save_data.index.astype(int)
                save_data.columns = pd.Series(save_data.columns).apply(lambda x: index_dict[x])

                save_data = pd.concat([old_data, save_data])
                save_data = save_data[~save_data.index.duplicated('last')]

                if save_data.index[0] > get_pre_trade_date(base_date, -1):
                    print(data_name + '数据存在问题，需要检查！！！！！！！！！！！！！！！！！！！！！！！')
                # 将该输出到保存地址中
                save_data = save_data.reindex(get_date_range(base_date, int(end_date)))
                save_data.to_hdf('%s/%s.h5' % (save_path,data_name), data_name, format='t')
                print(data_name + '存储完毕')
            else:
                print(data_name +'今日数据已更新')

            gc.collect()

# 3、存储行业基础数据：要把所有字符串日期，改为数字日期，初始日期变成特定日期20100101
save_inddaily_dict = {
        'ASWSIndexEOD': {'sw_open': 'S_DQ_OPEN', 'sw_high': 'S_DQ_HIGH', 'sw_low': 'S_DQ_LOW','sw_close': 'S_DQ_CLOSE',
                         'sw_pre_close': 'S_DQ_PRECLOSE', 'sw_volume': 'S_DQ_VOLUME', 'sw_amt': 'S_DQ_AMOUNT',
                         'sw_pe':'S_VAL_PE','sw_pb':'S_VAL_PB'},
        'AIndexIndustriesEODCITICS':{'zx_open': 'S_DQ_OPEN', 'zx_high': 'S_DQ_HIGH', 'zx_low': 'S_DQ_LOW','zx_close': 'S_DQ_CLOSE',
                         'zx_pre_close': 'S_DQ_PRECLOSE', 'zx_volume': 'S_DQ_VOLUME', 'zx_amt': 'S_DQ_AMOUNT'},
    }

def get_ind_factor(save_data_dict, start= base_date, resave=False):
    save_path = base_address + 'dailyBench/'
    # resave = True：进行历史全周期数据的重刷使用  False：添加到该结果的后面
    end_date = str(get_recent_trade_date())  # 1、获取数据都区范围
    # 2、确认路径是否存在，不存在创建路径
    os.makedirs(save_path) if os.path.exists(save_path) == False else None
    # 3、申万数据和中信数据分开来保存
    for table in save_data_dict.keys():
        for data_name in save_data_dict[table].keys():
            if data_name[:2] == 'sw':
                ind_list1 = get_ind_con('SW', level=[1, 2, 3])
                ind_list2 = get_ind_con('SW2021', level=[1, 2, 3])
                ind_name = list(set(ind_list1).union(set(ind_list2)))
            elif data_name[:2] == 'zx':
                ind_list = get_ind_con('CITICS', level=[1, 2, 3])
                ind_name = list(ind_list.keys())

            old_name = save_data_dict[table][data_name]
            data_str = old_name + ', TRADE_DT, S_INFO_WINDCODE'
            # （1）确定数据的提取日期，即start_date；如果不进行数据冲刷，且存储路径该结果存在，则调整start_date
            if (resave == False) & (os.path.exists(save_path + data_name + '.h5') == True):
                old_data = pd.read_hdf(save_path + data_name + '.h5', data_name)
                start_date = str(old_data.index[-1])
            else:
                old_data = pd.DataFrame()
                start_date = start
            # （2）获取信息
            if int(start_date) < int(end_date):
                sql = r"select %s from wind.%s a where a.TRADE_DT >= '%s' and  a.TRADE_DT <= '%s' " % \
                      (data_str, table, start_date, end_date)
                data_values = pd.read_sql(sql, con)
                data_values = data_values[data_values['S_INFO_WINDCODE'].isin(ind_name)]

                save_data = data_values.pivot_table(values=old_name, index='TRADE_DT', columns='S_INFO_WINDCODE')
                # 转换数据格式
                save_data.index = save_data.index.astype(int)
                save_data = pd.concat([old_data, save_data])
                save_data = save_data[~save_data.index.duplicated('last')]

                if save_data.index[0] > get_pre_trade_date(base_date, -1):
                    print(data_name + '数据存在问题，需要检查！！！！！！！！！！！！！！！！！！！！！！！')
                # 将该输出到保存地址中
                save_data = save_data.reindex(get_date_range(base_date, int(end_date)))
                save_data.to_hdf('%s/%s.h5' % (save_path,data_name), data_name, format='t')
                print(data_name + '存储完毕')
            else:
                print(data_name +'今日数据已更新')

            gc.collect()


# 4、储存必要的基础数据的衍生数据
def get_other_factor():
    address = base_address + 'daily/'
    # 1、保存全市场股票池
    _store_stock_list(address)
    # 2、保存ST个股
    _store_ST(address)
    # 3、保存停牌个股
    _store_live_days_and_pause(address)
    # 4、保存涨跌停个股
    _store_price_get_limit(address)
# (1)存储基础股票池数据
def _get_stock_list(date,address):

    _date = _check_input_date(date)
    df = pd.read_hdf('%s/pre_close.h5' % address, 'pre_close')
    df.index = df.index.map(int)
    df.columns = df.columns.map(trans_windcode2int)
    df = df.stack().reset_index()
    df.columns = ['date', 'code', 'true']
    df = df[['date', 'code']]
    return df
def _store_stock_list(address):
    date = get_date_range(20100101)
    df = _get_stock_list(date,address)
    df['true'] = True
    df = df.pivot('date', 'code', 'true').fillna(False)
    df = df.apply(pd.to_numeric,errors='ignore')
    df.to_hdf('%s/stock_list.h5' % address, 'stock_list', format='t')

# (2)保存特别处理股票（ST，PT，L-退市，T-退市）
def judge_ST():
    ST_table = 'AShareST'
    ST_value = ['S_INFO_WINDCODE', 'ENTRY_DT', 'REMOVE_DT', 'S_TYPE_ST', 'ANN_DT']
    ST_value_str = re.sub('[\'\[\]]','',str(ST_value))

    sql = r"select %s from wind.%s" % (ST_value_str, ST_table)
    ST = pd.read_sql(sql, con)
    ST.columns = ['code', 'dateIn', 'dateOut', 'type', 'dateAnn']
    ST = ST[ST['code'].map(lambda x: x[0]).isin(['0', '3', '6'])] # 消除非A股的个股
    ST['code'] = ST['code'].map(lambda x: int(x[:6]))
    ST = ST.sort_values('dateAnn')

    st = ST[['code', 'dateIn', 'dateOut']].copy()
    st['value'] = 1
    stEntry = st.pivot('dateIn', 'code', 'value')
    stRemove = st[['dateOut', 'code', 'value']].dropna().drop_duplicates().pivot('dateOut', 'code', 'value')
    st = stEntry.sub(stRemove, fill_value=0).replace(0, np.nan).ffill() > 0.5
    st.index = st.index.map(int)
    return st
def _store_ST(address):
    date = get_date_range(20100101)
    ST = judge_ST().reindex(date).ffill()
    stock_list = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list')

    ST = ST.reindex_like(stock_list) == 1
    ST = ST.apply(pd.to_numeric,errors='ignore')
    ST.to_hdf('%s/ST.h5' % address, 'ST', format='t')
# (3)保存停牌个股，给个股上市天数
def _store_live_days_and_pause(address):
    amt = pd.read_hdf('%s/amt.h5' % address, 'amt')
    amt.index = amt.index.map(int)
    amt.columns = amt.columns.map(trans_windcode2int)

    pause = amt.fillna(0) <= 1
    live_days = (~pause).cumsum()
    stock_list = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list')
    live_days = live_days.reindex_like(stock_list).fillna(0)
    pause = pause.reindex_like(stock_list) == 1
    live_days = live_days.apply(pd.to_numeric, errors='ignore')
    pause = pause.apply(pd.to_numeric, errors='ignore')

    live_days.to_hdf('%s/live_days.h5' % address, 'live_days', format='t')
    pause.to_hdf('%s/pause.h5' % address, 'pause', format='t')
# (4)获取涨跌停个股
def _store_price_get_limit(address):
    date = get_date_range(20100101)
    _date = _check_input_date(date)
    maxupordown  = pd.read_hdf('%s/limit_state.h5' % address, 'limit_state')

    limit_up = maxupordown > 0.5
    limit_down = maxupordown < -0.5

    stock_list = pd.read_hdf('%s/stock_list.h5' % address, 'stock_list')
    limit_up = limit_up.reindex_like(stock_list) == 1
    limit_down = limit_down.reindex_like(stock_list) == 1

    limit_up = limit_up.apply(pd.to_numeric, errors='ignore')
    limit_down = limit_down.apply(pd.to_numeric, errors='ignore')
    limit_up.to_hdf('%s/limit_up.h5' % address, 'limit_up', format='t')
    limit_down.to_hdf('%s/limit_down.h5' % address, 'limit_down', format='t')


if __name__ == '__main__':
    if (datetime.date.today().day >=28) or (os.path.exists(base_address+'daily/') == False): # 月末更新一下
        get_list_factor()

    from dataApi.tradeDate import _check_input_date, get_pre_trade_date, get_date_range
    from dataApi.stockList import *

    now_time = time.time()
    get_stock_factor(save_stockdaily_dict)  # 保存个股日品信息
    get_bench_factor(save_benchdaily_dict)    # 保存指数日频信息
    get_ind_factor(save_inddaily_dict)      # 保存行业日频信息
    print('基础数据更新完毕：',str(round((time.time() - now_time)/60,3))+'分钟')
    get_bench_weight()
    get_ind_weight()
    print('指数权重数据更新完毕：',str(round((time.time() - now_time)/60,3))+'分钟')

    get_other_factor()
    print('全部数据更新完毕：',str(round((time.time() - now_time)/60,3))+'分钟')


