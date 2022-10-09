import time
from BarraFactor.barra_factor import *
from dataApi.dividend import *
from dataApi.getData import *
from tqdm import tqdm
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
from SectorRotation.FactorTest import *
from usefulTools import *

def trans_factor_to_industry(df,code_ind,ind_name):
    df_copy = df.copy()
    for i in [0, 1, 2, 3]:
        df.iloc[i::4] = df.iloc[i::4].ffill(limit=2)

    save_index = df_copy.index
    df_copy.index = pd.Series(df_copy.index).apply(lambda x: get_recent_trade_date(x))
    # 进行行业合并
    new_code_ind = code_ind.loc[df_copy.index]
    new_df = pd.concat([df_copy[(new_code_ind == x).replace(False,np.nan).bfill(limit = 16).replace(np.nan,False)].mean(axis=1).rename(x) for x in ind_name], axis=1)#[ind_useful]
    new_df.index = save_index

    new_df.replace(0,np.nan,inplace=True)

    return new_df.astype(float).round(5)

def Factor_new(start_date,end_date,ind,period='M'):

    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date, period=period)
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list).reindex(period_date_list)[stock_pool]
    ind_useful = get_useful_ind(ind, date_list)
    # 指标
    table = 'AShareMoneyFlow'
    factor_str = 'S_INFO_WINDCODE,TRADE_DT, BUY_VALUE_EXLARGE_ORDER,SELL_VALUE_EXLARGE_ORDER,' \
                 'BUY_VALUE_SMALL_ORDER, SELL_VALUE_SMALL_ORDER'
    big_order_in = pd.DataFrame()
    big_order_out = pd.DataFrame()
    small_order_in = pd.DataFrame()
    small_order_out = pd.DataFrame()
    for i in tqdm(range(1,len(period_date_list))):
        start = period_date_list[i-1]
        end = period_date_list[i]
        sql = r"select %s from wind.%s a where a.TRADE_DT >= '%s' and a.TRADE_DT <= '%s'" %\
              (factor_str, table, str(date_list[0]),str(date_list[-1]))
        data_values = pd.read_sql(sql, con)

        big_order_in_month = data_values.pivot_table(values='BUY_VALUE_EXLARGE_ORDER', index='TRADE_DT', columns='S_INFO_WINDCODE')
        big_order_out_month = data_values.pivot_table(values='SELL_VALUE_EXLARGE_ORDER', index='TRADE_DT', columns='S_INFO_WINDCODE')
        small_order_in_month = data_values.pivot_table(values='BUY_VALUE_SMALL_ORDER', index='TRADE_DT', columns='S_INFO_WINDCODE')
        small_order_out_month = data_values.pivot_table(values='SELL_VALUE_SMALL_ORDER', index='TRADE_DT', columns='S_INFO_WINDCODE')

        big_order_in = pd.concat([big_order_in,big_order_in_month])
        big_order_out = pd.concat([big_order_out, big_order_out_month])
        small_order_in = pd.concat([small_order_in, small_order_in_month])
        small_order_out = pd.concat([small_order_out, small_order_out_month])







    factor = ind_factor[ind_useful].loc[start_date:end_date]

    return factor


start_date, end_date,ind = 20150101,20220928,'SW1'
factor = factor_test(Factor_new,start_date,end_date,ind)

factor_name = 'Factor_new'
test_result, value_result, ic, rank_ic = factor_in_box(factor,factor_name, ind,fee=0.001)
test_result
value_result[[1,2,3,4,5]]

# 检查一下所有的因子，是否具有np.inf和-np.inf


def Factor_new1(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    #net_profit = get_quarter_1factor('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type='408001000', date_list=date_list).dropna(
    #    how='all', axis=1)
    curr_asset = get_quarter_1factor('TOT_CUR_ASSETS', 'AShareBalanceSheet', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)
    inventory = get_quarter_1factor('INVENTORIES', 'AShareBalanceSheet', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)
    prepay = get_quarter_1factor('PREPAY', 'AShareBalanceSheet', report_type='408001000',
                                  date_list=date_list).dropna(how='all', axis=1)

    curr_liab = get_quarter_1factor('TOT_CUR_LIAB', 'AShareBalanceSheet', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)
    #opr_cost = get_quarter_1factor('TOT_OPER_COST', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)

    report_data = (curr_asset - inventory.replace(np.nan,0) - prepay.replace(np.nan,0)).round(5) / curr_liab.replace(0,np.nan)

    save_index = report_data.index
    report_data.index = pd.Series(report_data.index).apply(lambda x: get_recent_trade_date(x))
    report_data = report_data[stock_pool]
    report_data.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        report_data.iloc[i::4] = report_data.iloc[i::4].ffill(limit=2)


    #yoy_diff = get_yoy(report_data)#.diff(1)
    #diff_expect = yoy_diff.rolling(8).mean().astype(float).round(5)
    #diff_std = yoy_diff.rolling(8).std().astype(float).round(5)

    #new_factor = (yoy_diff - diff_expect) / diff_std

    #new_factor = report_data.replace(0, np.nan).diff().round(5) / abs(report_data.replace(0, np.nan).shift(1))
    new_factor = report_data.diff()
    #new_factor = new_factor.T.apply(lambda x: mad(x)).T
    #new_factor = report_data.replace(0, np.nan).diff().round(5)
    new_factor = new_factor.T.apply(lambda x:mad(x)).T

    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]

    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[start_date:end_date]

    factor = deal_data(factor) ** 2

    return factor


start_date, end_date,ind = 20150101,20220928,'SW1'
factor = factor_test(Factor_new,start_date,end_date,ind)

factor_name = 'Factor_new1'
test_result, value_result, ic, rank_ic = factor_in_box(factor,factor_name, ind,fee=0.001)
test_result



start_date, end_date,ind = 20150101,20220928,'SW1'
factor = factor_test(Factor_epredict_eps_1m,start_date,end_date,ind)

factor_name = 'Factor_epredict_eps_1m'
test_result, value_result, ic, rank_ic = factor_in_box(factor,factor_name, ind,fee=0.001)
test_result
value_result[[1,2,3,4,5]]