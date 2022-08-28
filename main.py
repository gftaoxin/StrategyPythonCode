import pandas as pd
import numpy as np
import cx_Oracle

con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True)

start_date = '20220722'
end_date= '20220815'
sql = r"select a.S_INFO_WINDCODE,a.TRADE_DT,a.S_DQ_CLOSE,a.S_DQ_ADJCLOSE from wind.AShareEODPrices a where a.TRADE_DT > '%s' and  a.TRADE_DT < '%s'" % (start_date,end_date)
A_close_data = pd.read_sql(sql, con)

# data = pd.concat([A_close_data, HK_close_data, NQ_close_data])

# data['TRADE_DT'] = pd.to_datetime([str(x)[:4] + "-" + str(x)[4:6] + "-" + str(x)[6:8] for x in data['TRADE_DT']])
# data.columns = ['stock_code', 'date', 'close', 'adj_close']



