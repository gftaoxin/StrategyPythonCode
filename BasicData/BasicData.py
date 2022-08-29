import os,datetime
import pickle

import pandas as pd
import numpy as np
import cx_Oracle
import gc,re

data_path = 'D:\Program\BasicData\\'
con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True) # 写入信号

# 存储股票代码，和存储交易日历，通常1个月更新一次即可，无需重复重新
def get_list_factor(save_path=data_path):
    # 确认路径是否存在，不存在创建路径
    os.makedirs(save_path) if os.path.exists(save_path) == False else print('存储路径已存在，继续运行')
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
    # 2、存储股票代码
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

get_list_factor(data_path)


#sql = r"select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE,S_DQ_ADJCLOSE from wind.AShareEODPrices a where a.TRADE_DT > '%s' and  a.TRADE_DT < '%s'" % (start_date, end_date)

# 注意：要把所有字符串日期，改为数字日期
# 这部分要进行数据检查：比如自动更新先获取最近的日期，然后看本地是否有该文件；有该文件则重新获取start_date
# 没有该文件让初始日期变成特定日期20130101

def get_df_factor(save_data_dict,start_date = '20130101',save_path=data_path,resave = False):
    # resave = True：进行历史全周期数据的重刷使用  False：添加到该结果的后面
    # 1、获取数据都区范围
    end_date = datetime.date.today().strftime("%Y%m%d")
    # 2、确认路径是否存在，不存在创建路径
    os.makedirs(save_path) if os.path.exists(save_path) == False else print('存储路径已存在，继续运行')
    # 3、针对每一个数据，进行存储，存储格式为单因子格式，date,code,factor
    for table in save_data_dict.keys():
        for data_name in save_data_dict[table].keys():
            old_name = save_data_dict[table][data_name]
            data_str = old_name +', TRADE_DT, S_INFO_WINDCODE'
            # （1）确定数据的提取日期，即start_date；如果不进行数据冲刷，且存储路径该结果存在，则调整start_date
            if (resave == False) & (os.path.exists(save_path + data_name + '.pkl') == True):
                old_data = pd.read_pickle(save_path + data_name + '.pkl')
                start_date = str(old_data.index[-1])
            else:
                old_data = pd.DataFrame()
            # （2）获取信息
            sql = r"select %s from wind.%s a where a.TRADE_DT > '%s' and  a.TRADE_DT < '%s'" % \
                  (data_str,table,start_date, end_date)
            data_values = pd.read_sql(sql, con)
            save_data = data_values.pivot_table(values=old_name, index='TRADE_DT', columns='S_INFO_WINDCODE')
            # 将该输出到保存地址中



            # 开始进行储存

                save_data.to_pickle(save_path +data_name +'.pkl')
            else:

                pd.concat([old_data,save_data],)


    # 2、股票日行情信息

    if save == True: # 进行全样本存储
        pre_close.to_pickle(data_path +'pre_close.pkl')

    else: # 进行样本添加
        pre_close = pd.read_pickle()

    gc.collect()

save_df_dict = {
    'AShareEODPrices': {
        'open':'S_DQ_PRECLOSE', 'high':'S_DQ_HIGH', 'low':'S_DQ_LOW', 'close':'S_DQ_CLOSE',
        'pre_close':'S_DQ_PRECLOSE', 'chg':'S_DQ_CHANGE', 'vol':'S_DQ_VOLUME', 'amt':'S_DQ_AMOUNT',
        'adj_close':'S_DQ_ADJCLOSE', 'adj_open':'S_DQ_ADJOPEN', 'adj_high':'S_DQ_ADJHIGH', 'adj_low':'S_DQ_ADJLOW',
        'adj_factor':'S_DQ_ADJFACTOR', 'vwap':'S_DQ_AVGPRICE',
        'limit_up':'S_DQ_LIMIT','limit_down':'S_DQ_STOPPING'
    },

    'AIndexEODPrices':{'open':'S_DQ_PRECLOSE', 'high':'S_DQ_HIGH', 'low':'S_DQ_LOW', 'close':'S_DQ_CLOSE',
                       'pre_close':'S_DQ_PRECLOSE','pct_chg':'S_DQ_PCTCHANGE','vol':'S_DQ_VOLUME', 'amt':'S_DQ_AMOUNT'}

}


save_list_dict = {
    'AShareCalendar': {'trade_date': 'TRADE_DAYS'},
    'AShareDescription':{'S_INFO_WINDCODE','S_INFO_CODE','S_INFO_NAME','S_INFO_LISTDATE','S_INFO_DELISTDATE'}
}



# data = pd.concat([A_close_data, HK_close_data, NQ_close_data])

# data['TRADE_DT'] = pd.to_datetime([str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8] for x in data['TRADE_DT']])
# data.columns = ['stock_code', 'date', 'close', 'adj_close']




# 1、股票日频信息


# 2、




# 2、指数日频信息




