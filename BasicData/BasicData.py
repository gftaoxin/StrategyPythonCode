import os
import pandas as pd
import numpy as np
import cx_Oracle
import gc,re

data_path = 'D:\Program\BasicData\\'

con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True) # 写入信号

sql = r"select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE,S_DQ_ADJCLOSE from wind.AShareEODPrices a where a.TRADE_DT > '%s' and  a.TRADE_DT < '%s'" % (start_date, end_date)

# 注意：要把所有字符串日期，改为数字日期
# 这部分要进行数据检查：比如自动更新先获取最近的日期，然后看本地是否有该文件；有该文件则重新获取start_date
# 没有该文件让初始日期变成特定日期20130101

# 创建一个git项目

def get_df_factor(start_date,end_date,save_data_dict,save_path=data_path,resave = False):
    start_date, end_date = str(start_date), str(end_date)
    if os.path.exists(data_path) == False: # 路径
        os.makedirs(data_path)
    # resave = True：进行历史全周期数据的重刷使用  False：添加到该结果的后面
    for table in save_data_dict.keys():
        data_list = set(save_data_dict[table].values())
        data_str = re.sub('[\'\{\}]','', str(data_list)) + ', TRADE_DT, S_INFO_WINDCODE'
        # 获取信息
        sql = r"select %s from wind.%s a where a.TRADE_DT > '%s' and  a.TRADE_DT < '%s'" % \
              (data_str,table,start_date, end_date)
        data_values = pd.read_sql(sql, con)
        # 每一项都输出到保存地址中
        for data_name in save_data_dict[table].keys():
            data_oldname = save_data_dict[table][data_name]
            save_data = data_values.pivot_table(values=data_oldname, index='TRADE_DT', columns='S_INFO_WINDCODE')
            # 开始进行储存
            if (save == True) or (os.path.exists(data_path +data_name +'.pkl')==False):
                save_data.to_pickle(data_path +data_name +'.pkl')
            else:
                old_data = pd.read_pickle(data_path +data_name +'.pkl')
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
}
start_date,end_date = 20130101,20220822



# data = pd.concat([A_close_data, HK_close_data, NQ_close_data])

# data['TRADE_DT'] = pd.to_datetime([str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8] for x in data['TRADE_DT']])
# data.columns = ['stock_code', 'date', 'close', 'adj_close']




# 1、股票日频信息


# 2、




# 2、指数日频信息




