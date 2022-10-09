import cx_Oracle,re
import pandas as pd

import getData
from dataApi.stockList import *
con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True) # 写入信号

def getEXRightDividend():
    table = 'AShareEXRightDividendRecord'
    factors = ['BONUS_SHARE_RATIO', 'CONVERSED_RATIO', 'RIGHTSISSUE_RATIO', 'SEO_RATIO', 'RIGHTSISSUE_PRICE',
               'SEO_PRICE', 'CASH_DIVIDEND_RATIO', 'EX_DATE', 'S_INFO_WINDCODE']
    factors_str = re.sub('[\'\[\]]', '', str(factors))

    sql = r"select %s from wind.%s a where a.EX_DATE >= '20100101'" % (factors_str, table)
    EXRightDividend = pd.read_sql(sql, con)

    EXRightDividend['shareRatio'] = EXRightDividend[['BONUS_SHARE_RATIO', 'CONVERSED_RATIO',
            'RIGHTSISSUE_RATIO', 'SEO_RATIO']].sum(axis=1)
    EXRightDividend['receiveRatio'] = pd.concat([EXRightDividend['RIGHTSISSUE_RATIO'] * EXRightDividend[
            'RIGHTSISSUE_PRICE'], EXRightDividend['SEO_RATIO'] * EXRightDividend['SEO_PRICE']], axis=1).sum(axis=1)
    EXRightDividend['payoutRatio'] = EXRightDividend['CASH_DIVIDEND_RATIO']
    EXRightDividend = EXRightDividend[['EX_DATE', 'S_INFO_WINDCODE', 'shareRatio', 'receiveRatio', 'payoutRatio']]
    EXRightDividend.columns = ['date', 'code', 'shareRatio', 'receiveRatio', 'payoutRatio']
    EXRightDividend['date'] = EXRightDividend['date'].map(int)
    EXRightDividend = EXRightDividend[EXRightDividend['code'].map(lambda x: x[0]).isin(['0','3','6'])]
    EXRightDividend['code'] = EXRightDividend['code'].map(lambda x: trans_windcode2int(x))
    EXRightDividend = EXRightDividend.sort_values('date')
    return EXRightDividend
