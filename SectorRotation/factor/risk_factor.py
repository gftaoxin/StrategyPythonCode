import time
from dataApi.dividend import *
from dataApi.getData import *
from BarraFactor.barra_factor import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from SectorRotation.FactorTest import *
from usefulTools import  *

# 风险指标1：成交额/市值的横向占比和纵向占比：如果当前市场整体成交额都处于高位，则不进行剥离
def RiskFactor_Amt(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=500), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    # 个股所属行业
    ind_close = get_daily_1factor('close', date_list=date_list, code_list=ind_name, type=ind[:-1])
    ind_pct20 = ind_close.pct_change(20,fill_method=None)
    ind_pct60 = ind_close.pct_change(60, fill_method=None)
    # 按行业累加起来
    mv = get_daily_1factor('mkt_free_cap', date_list=date_list)[stock_pool]
    amt = get_daily_1factor('amt', date_list=date_list)[stock_pool]

    amt_mv_ratio = pd.DataFrame(index=date_list, columns=ind_name)
    for i in ind_name:
        factor_i = amt[code_ind == i].sum(axis=1) / mv[code_ind == i].sum(axis=1)
        amt_mv_ratio[i] = factor_i

    amt_ratio = amt_mv_ratio.rolling(20).sum()
    history_amt_ratio = ts_rank(amt_ratio,rol_day=240)

    hot_ind = (history_amt_ratio > 0.9) & (amt_ratio.rank(pct=True,axis=1) > 0.9) & \
              ((ind_pct20.rank(pct=True,axis=1) > 0.8) | (ind_pct60.rank(pct=True,axis=1) > 0.8))

    hot_ind.loc[history_amt_ratio[history_amt_ratio.mean(axis=1) > 0.9].index] = False
    risk_factor = hot_ind[ind_useful].loc[start_date:end_date]
    risk_factor = risk_factor.fillna(False)

    return risk_factor

# 风险指标2：换手率处于高位 & 波动率处于高位。如果当前市场整体换手率都比较高，则不进行剥离
def RiskFactor_Turn(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=500), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    # 个股所属行业
    bench_close= get_daily_1factor('close', date_list=date_list,type='bench')['wind_A']
    ind_close = get_daily_1factor('close', date_list=date_list, code_list=ind_name, type=ind[:-1])
    ind_pct = ind_close.pct_change(1, fill_method=None)

    ind_20pct = ind_close.pct_change(20, fill_method=None)
    ind_60pct = ind_close.pct_change(60, fill_method=None)

    ind_std20 = ind_pct.rolling(60).std()
    bench_std = bench_close.pct_change(fill_method=None).rolling(60).std()

    # 按行业累加起来
    vol = get_daily_1factor('volume', date_list=date_list)[stock_pool]
    shares = get_daily_1factor('free_float_shares', date_list=date_list)[stock_pool]

    ind_turn = pd.DataFrame(index=date_list, columns=ind_name)
    for i in ind_name:
        ind_code = vol[code_ind == i].sum(axis=1) / shares[code_ind == i].sum(axis=1)
        ind_turn[i] = ind_code

    ind_std = (ind_std20.T / bench_std).T
    histroy_std = ts_rank(ind_std,rol_day=240)

    # 当前换手率处于历史高位
    turn_rol20 = ind_turn.rolling(20).mean()
    history_turn_ratio = ts_rank(turn_rol20, rol_day=240)

    hot_ind = (history_turn_ratio > 0.8) & (turn_rol20.rank(pct=True,axis=1) > 0.9) & (histroy_std > 0.6) & \
              ((ind_20pct.rank(pct=True,axis=1) > 0.6) | (ind_60pct.rank(pct=True,axis=1) > 0.6))
    hot_ind.loc[history_turn_ratio[history_turn_ratio.mean(axis=1) > 0.8].index] = False
    risk_factor = hot_ind[ind_useful].loc[20160101:end_date]
    risk_factor = risk_factor.fillna(False)

    return risk_factor

# 风险指标3：分化风险
def RiskFactor_Corr(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=500), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)

    # 个股所属行业
    ind_close = get_daily_1factor('close', date_list=date_list, code_list=ind_name, type=ind[:-1])
    ind_pct = ind_close.pct_change(1, fill_method=None)
    ind_amt = get_daily_1factor('amt', date_list=date_list, code_list=ind_name, type=ind[:-1])

    price_amt_corr = rolling_corr(ind_pct, ind_amt,window=60)  # 行业自身成交额，和行业成交量的量价相关性
    ind_20pct = ind_close.pct_change(20, fill_method=None)

    # 行业内个股之间（个股和行业）的量价相关性
    close = get_daily_1factor('close', date_list=date_list)[stock_pool]
    code_pct = close.pct_change(1, fill_method=None)

    ind_code_corr = pd.DataFrame(index=date_list, columns=ind_name)
    for i in ind_name:
        ind_code_pct = code_pct[code_ind == i].dropna(how='all',axis=1)
        ind_ind_pct =  pd.concat([ind_pct[i] for x in ind_code_pct.columns],axis=1)

        ind_code_corr[i] = rolling_corr(ind_code_pct, ind_ind_pct,window=60).mean(axis=1)

    # 开始组合
    history_corr_ratio = ts_rank(price_amt_corr, rol_day=240)
    history_ind_scorr_ratio = ts_rank(ind_code_corr, rol_day=240)
    '''
    # 板块的量价相关性很高（历史+时序），但不是市场最强的（不是往上冲而是回落），且分化度还很高，那么很容易回落
    hot_ind1 = (history_corr_ratio > 0.9) & (price_amt_corr.rank(pct=True, axis=1) > 0.9) & (ind_20pct.rank(pct=True,axis=1) < 0.5) & \
              (history_ind_scorr_ratio < 0.5) #& (history_ind_scorr_ratio.rank(pct=True,axis=1) < 0.3)
    risk_factor1 = hot_ind1[ind_useful].loc[20160101:end_date]
    risk_factor1 = risk_factor1.fillna(False)
    '''
    # 板块自身的量价相关性很低
    hot_ind2 = (history_corr_ratio < 0.1) & (price_amt_corr.rank(pct=True, axis=1) < 0.1) & (history_ind_scorr_ratio <0.7) & \
              (history_ind_scorr_ratio.rank(pct=True, axis=1) > 0.7) & (ind_20pct.rank(pct=True,axis=1) < 0.7)
    risk_factor2 = hot_ind2[ind_useful].loc[20160101:end_date]
    risk_factor2 = risk_factor2.fillna(False)

    #risk_factor = risk_factor1 | risk_factor2
    # return factor

    return risk_factor2




start_date, end_date,ind = 20151231,20221110,'SW1'
save_path = 'E:/FactorTest/risk_factor/'

risk_amtfactor = RiskFactor_Amt(start_date,end_date,ind)
risk_turnfactor = RiskFactor_Turn(start_date,end_date,ind)
risk_corr = RiskFactor_Corr(start_date,end_date,ind)

risk_amtfactor.to_pickle(save_path + 'risk_amtfactor' + '.pkl')
risk_turnfactor.to_pickle(save_path + 'risk_turnfactor' + '.pkl')
risk_corr.to_pickle(save_path + 'risk_corr' + '.pkl')



'''
risk1 = risk_amtfactor.copy()
risk2 = risk_turnfactor.copy()
risk3 = risk_corr.copy()

risk4 = risk1 | risk2 | risk3
risk5 =  (risk1) & (~risk2) & (~risk3)
risk6 =  (risk2) & (~risk1) & (~risk3)
risk7 =  (risk3) & (~risk1) & (~risk2)


start_date, end_date,ind = 20151231,20221102,'SW1'
self = FactorTest(test_start_date=20150101, test_end_date=20221028, ind=ind, day=20,fee=0.001)

result = self.risk_factor_test(risk4,future_period='day')
print(len(result))
print(-result[result['win_rate']>0]['excess'].mean() / result[result['win_rate']==0]['excess'].mean())
'''
