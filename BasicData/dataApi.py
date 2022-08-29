import os,datetime
import pickle
import pandas as pd
import numpy as np
import cx_Oracle
import gc,re



def get_trade_date(start_date,end_date):
    trade_list = pd.read_pickle(data_path + 'trade_date.pkl')
    trade_list>=start_date

    trade_list.index(start_date)


    pass

# 函数1.1：获取最近的交易日
def get_recent_trade_date():
    pass


# 函数2：获取股票列表
def get_code_list():
    pass











