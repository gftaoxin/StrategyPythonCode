import os,datetime
import pandas as pd
import numpy as np
import cx_Oracle
import gc,re

base_address = 'E:/ABasicData/'
base_date = '20100101'

'''
['open', 'S_DQ_OPEN', '开盘价']



, 'high': 'S_DQ_HIGH', 'low': 'S_DQ_LOW', 'close': 'S_DQ_CLOSE',
'pre_close': 'S_DQ_PRECLOSE', 'pct_chg': 'S_DQ_PCTCHANGE', 'vol': 'S_DQ_VOLUME', 'amt': 'S_DQ_AMOUNT',
'adj_close': 'S_DQ_ADJCLOSE', 'adj_open': 'S_DQ_ADJOPEN', 'adj_high': 'S_DQ_ADJHIGH',
'adj_low': 'S_DQ_ADJLOW',
'adj_factor': 'S_DQ_ADJFACTOR', 'vwap': 'S_DQ_AVGPRICE',
'limit_up': 'S_DQ_LIMIT', 'limit_down': 'S_DQ_STOPPING'
'''