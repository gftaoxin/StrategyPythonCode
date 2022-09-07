import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np


# tradeData模块测试
trade_dates,trade_dates[0],trade_dates[-1]
trade_weeks,trade_weeks[0],trade_weeks[-1]
trade_months,trade_months[0],trade_months[-1]
trade_quarters,trade_quarters[0],trade_quarters[-1]
trade_half_years,trade_half_years[0],trade_half_years[-1]
trade_years,trade_years[0],trade_years[-1]
report_dates,report_dates[0],report_dates[-1]

get_today(15),get_today(8)

get_recent_trade_date(20220905,period='D'),get_recent_trade_date(20220905,period='Y')

get_recent_trade_date(20220905,period='W')
get_pre_trade_date(20220905,offset=-1,period='W')

get_date_range(20140101,20220901,period='W')[0]

get_trade_date_interval(20220907,base_date=20220905)

trans_datetime2int(trans_int2datetime(202212310624,input_date=True,out_format='m',time_digit=4))

# stockList模块测试
trans_windcode2int(1)
get_stock_list()
len(get_stock_list(20151231))
len(get_code_list())

a = clean_stock_list(stock_list='HS300',no_ST=True,least_live_days=240,no_pause=True,least_recover_days=1)
a.sum(axis=1)


# getData测试模块
stock_factor_list = [
    'stock_list', 'CITICS1',  'CITICS2', 'SW1', 'SW2', 'SW3', 'SW20211', 'SW20212', 'SW20213',
    'SZ50', 'HS300', 'ZZ500', 'ZZ800',
    'net_profit_parent_comp_ttm','net_profit_parent_comp_lyr',
    'net_cash_flows_oper_act_ttm','net_cash_flows_oper_act_lyr',
    'oper_rev_ttm','oper_rev_lyr','net_incr_cash_cash_equ_ttm','pause','live_days','pre_close',
    'open','high','low','close','pre_close_badj','open_badj','high_badj','low_badj','close_badj',
    'pct_chg','vwap','adjfactor','amt','volume','turn','free_turn','total_shares','net_incr_cash_cash_equ_lyr',
    'pe','pe_ttm','pb','pcf_ocf','pcf_ocf_ttm','pcf_ncf','pcf_ncf_ttm','ps','ps_ttm','net_assets_today',
    'limit_state','limit_up','limit_down','A_shares','float_A_shares','free_float_shares','mkt_cap_ard',
    'mkt_free_cap','deal_num','beta_day_1y','beta_100w','beta_24m','beta_60m','alpha_day_1y','alpha_100w',
    'alpha_24m','alpha_60m']

bench_factor_list = [
    'open','high','low','close','pre_close','pct_chg','turn','free_turn','volume','amt','pe','pe_ttm',
    'pb','pcf','pcf_ttm','ps','ps_ttm','dividend_yield','peg_his']

SW_factor_list = ['open','high','low','close','pre_close','volume','amt','pe','pb']
CITICS_factor_list = ['open','high','low','close','pre_close','volume','amt']

get_daily_1day(stock_factor_list,date = 20220905,type='stock')
get_daily_1day(bench_factor_list,date = 20220905,type='bench')
get_daily_1day(SW_factor_list,date = 20220905,type='SW')
get_daily_1day(CITICS_factor_list,date = 20220905,type='CITICS')


get_daily_1stock(code=1,factor_list=stock_factor_list,date_list=get_date_range(20150101,20220905),type='stock')
get_daily_1stock(code='HS300',factor_list=bench_factor_list,date_list=get_date_range(20150101,20220905),type='bench')
get_daily_1stock(code='801010.SI',factor_list=SW_factor_list,date_list=get_date_range(20150101,20220905),type='SW')
get_daily_1stock(code='CI005028.WI',factor_list=CITICS_factor_list,date_list=get_date_range(20150101,20220905),type='CITICS')

for factor in stock_factor_list:
    t =time.time()
    print(get_daily_1factor(factor,date_list = get_date_range(20150101,20220905), type='stock'))
    print(factor,time.time()-t)

for factor in bench_factor_list:
    t = time.time()
    print(get_daily_1factor(factor, date_list=get_date_range(20150101, 20220905), type='bench'))
    print(factor,time.time() - t)

for factor in SW_factor_list:
    t = time.time()
    print(get_daily_1factor(factor, date_list=get_date_range(20150101, 20220905), type='SW'))
    print(factor, time.time() - t)

for factor in CITICS_factor_list:
    t = time.time()
    print(get_daily_1factor(factor, date_list=get_date_range(20150101, 20220905), type='CITICS'))
    print(factor, time.time() - t)


get_quarter_1factor(factor='STOT_CASH_INFLOWS_OPER_ACT',table='AShareCashFlow',report_type='408001000',date_list=get_date_range(20150101,20210101))

df  = get_single_quarter(factor = 'OPER_REV',table='AShareIncome')

get_ttm_quarter(factor = 'OPER_REV',table='AShareIncome')

get_qoq(df)
get_yoy(df)

fill_quarter2daily_by_fixed_date(df, keep = 'first')
fill_quarter2daily_by_issue_date(df, 'AShareIncome', '408002000', keep = 'first')











tables = 'GFZQEDB'
factor_list = ['F2_4112','F3_4112','F4_4112','F5_4112','TDATE','INDICATOR_NUM']
factor_list_str = re.sub('[\'\[\]]', '', str(factor_list))

sql =r"select %s from wind.%s a where a.TDATE >= '%s'" % (factor_list_str, tables, '20100101')
save_data = pd.read_sql(sql, con)

save_bydate = save_data[save_data['F5_4112']==1]
save_byweek = save_data[save_data['F5_4112']==2]
save_bymonth = save_data[save_data['F5_4112']==3]
save_byquarters = save_data[save_data['F5_4112']==4]
save_byyear = save_data[save_data['F5_4112']==6]

save_data1 = save_byyear.pivot_table('INDICATOR_NUM',index='TDATE',columns = 'F3_4112')
save_data




save_data['INDICATOR_NUM']
