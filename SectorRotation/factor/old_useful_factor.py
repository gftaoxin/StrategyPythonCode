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

def Factor_Dragon(start_date,end_date,ind):
    # 逻辑：将板块内的个股拆分为成交量大的个股，和成交量小的个股；关注版块内成交量比较大的个股，和板块内成交量比较小的个股，走势的差异。
    # 在同样的市场行情下，走势差异度最大的板块，和走势差异度最小的板块；未来表现会弱于走势差异度平均的板块。
    date_list = get_date_range(get_pre_trade_date(start_date,offset=120),end_date)
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())
    # 个股所属行业
    code_ind = get_daily_1factor(ind, date_list=date_list)
    amt = get_daily_1factor('amt', date_list=date_list)
    amt_20days = amt.rolling(20).mean()

    close_badj = get_daily_1factor('close_badj', date_list=date_list)
    pct_20days = close_badj.pct_change(20)
    # 因子：
    # 龙头股：行业内最近20日成交金额占比前60%的成分股
    # 普通股：其余个股
    # 分歧度因子：龙头股20日收益率 - 普通股20日收益率
    dragon_factor = pd.DataFrame(index=date_list,columns=ind_name)

    for i in ind_name:
        dragon_pct = pct_20days[amt_20days[code_ind == i].rank(pct=True,axis=1) >=0.4].mean(axis=1)
        #other_pct = pct_20days[amt_20days[code_ind == i].rank(pct=True, axis=1) < 0.4].mean(axis=1)
        other_pct = pct_20days[code_ind == i].mean(axis=1)

        dragon_factor[i] = (dragon_pct - other_pct)

    dragon_factor =  (dragon_factor.sub(dragon_factor.mean(axis=1),axis=0).div(dragon_factor.std(axis=1),axis=0)) ** 2

    # 给结果根据行业的情况赋值
    ind_close = get_daily_1factor('close', date_list=date_list, type=ind[:-1])[ind_name]
    dragon_factor =dragon_factor[~np.isnan(ind_close)]



    return -dragon_factor.loc[get_pre_trade_date(start_date,offset=1):end_date]

def Factor_NorthAmtMv(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=200), end_date)
    north_volume = get_daily_1factor('north_quantity',date_list=date_list).dropna(how='all')
    close = get_daily_1factor('close',date_list=date_list).dropna(how='all')
    north_amt = (abs(north_volume.diff(1)) * close).dropna(how='all')

    # 个股所属行业
    code_ind = get_daily_1factor(ind,date_list=date_list)
    amt = get_daily_1factor('amt',date_list=date_list)
    free_mv =get_daily_1factor('mkt_free_cap',date_list=date_list)
    # 因子1：北向资金行业累计净流入金额/行业在北向资金内个股的累计成交额
    ind_name = list(get_ind_con(ind_type=ind[:-1],level=int(ind[-1])).keys())

    north_ind_amt = pd.concat([free_mv[code_ind == x].sum(axis=1).rename(x) for x in ind_name],axis=1).dropna(how='all')
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name],axis=1)

    factor = (north_ind_amt_in / (north_ind_amt)).rolling(60).sum()
    ind_df = (factor).dropna(how='all')

    return ind_df.loc[get_pre_trade_date(start_date,offset=1):end_date]

def Factor_ResidualMometumn_3barra(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date, period=period)
    # 个股log(行业市值)
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())
    # ind_mv = get_modified_ind_mv(date_list=date_list, ind_type=ind).loc[period_date_list].iloc[1:]

    size = deal_data(Barra_size(get_pre_trade_date(start_date, offset=300), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)
    btop = deal_data(Barra_BTOP(get_pre_trade_date(start_date, offset=300), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)
    #beta = deal_data(Barra_Beta(get_pre_trade_date(start_date, offset=300), end_date, ind).reindex(period_date_list).astype(float)).fillna(0)
    pe = deal_data(PE_TTM(get_pre_trade_date(start_date, offset=300), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)

    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1]).loc[period_date_list].dropna(how='all')
    pct_20days = ind_close.pct_change()
    # 开始进行时序回归
    factor = pd.DataFrame(index=period_date_list,columns=ind_name)

    for i in ind_name:
        df_y = pct_20days[[i]].dropna(how='all').astype(float)
        df_x = pd.concat([size[i].rename('size').dropna(),btop[i].rename('btop').dropna(),
                          pe[i].rename('pe')],axis=1).dropna().astype(float)
        common_date = sorted(list(set(df_y.index).intersection(df_x.index)))
        regression_residual = ts_rolling_regression(df_x.loc[common_date],df_y.loc[common_date],rolling_days=12)[1]
        factor[i] = regression_residual[i]

    factor = factor.loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float).dropna(how='all')
    factor = factor[~np.isnan(ind_close)]

    return factor

def Factor_NorthMaxdivPositve(start_date,end_date,ind):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=200), end_date)
    north_volume = get_daily_1factor('north_quantity', date_list=date_list).dropna(how='all')
    close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
    north_amt = (north_volume.diff(1) * close).dropna(how='all')

    # 个股所属行业
    code_ind = get_daily_1factor(ind, date_list=date_list)
    # 因子1：北向资金行业累计净流入金额/行业在北向资金内个股的累计成交额
    ind_name = list(get_ind_con(ind_type=ind[:-1], level=int(ind[-1])).keys())
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)

    north_ind_positive_in = north_ind_amt_in.copy()
    north_ind_positive_in[north_ind_positive_in < 0] = 0

    factor = north_ind_amt_in.rolling(120).max() / north_ind_positive_in.rolling(120).sum()
    ind_df = (factor).dropna(how='all')

    return ind_df.loc[get_pre_trade_date(start_date, offset=1):end_date]

def Factor_SUE(start_date,end_date,ind,period='M'):
    # 逻辑：SUE盈余惊喜，当期SUE - 过去8期SUE / 过去8期SUE的标准差
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1,no_ST=False,no_pause=False, no_limit_up=False, start_date=date_list[0],end_date=period_date_list[-1]).reindex(period_date_list).dropna(how='all')

    earning = get_quarter_1factor('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all',axis=1)
    save_index = earning.index
    earning.index = pd.Series(earning.index).apply(lambda x:get_recent_trade_date(x))
    earning = earning[stock_pool]
    earning.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0,1,2,3]:
        earning.iloc[i::4] = earning.iloc[i::4] .ffill()

    earnings_yoy_diff = get_yoy(earning)
    expect_earning = earnings_yoy_diff.rolling(4).mean()
    std_earning = earnings_yoy_diff.rolling(4).std()

    SUE = (earnings_yoy_diff - expect_earning) / std_earning
    SUE = fill_quarter2daily_by_fixed_date(SUE,keep='last').reindex(period_date_list).dropna(how='all')
    SUE = SUE[stock_pool]
    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[SUE.index]

    ind_SUE = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        SUE_i = (SUE[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1),axis=0)).sum(axis=1)
        ind_SUE[i] = SUE_i

    factor = ind_SUE[ind_useful].loc[start_date:end_date]

    return factor

def Factor_12and1Momentum(start_date,end_date,ind,period='M'):
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date, period=period)
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    # 个股所属行业
    ind_close =get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1]).reindex(period_date_list)
    pct_12m = ind_close.pct_change(12,fill_method=None)
    pc_1m = ind_close.pct_change(1,fill_method=None)

    factor = (pct_12m + pc_1m)[ind_useful].loc[start_date:end_date].astype(float)

    return factor

if __name__ == '__main__':
    start_date = 20150101
    end_date = 20211130
    ind = 'SW1'
    save_path = 'E:/FactorTest/old_useful_factor/'
    Factor_Dragon(start_date, end_date, ind).to_pickle(save_path + 'Factor_Dragon' + '.pkl')
    Factor_NorthAmtMv(start_date,end_date,ind).to_pickle(save_path + 'Factor_NorthAmtMv' + '.pkl')
    Factor_ResidualMometumn_3barra(start_date,end_date,ind,period='M').to_pickle(save_path + 'Factor_ResidualMometumn_3barra' + '.pkl')
    Factor_NorthMaxdivPositve(start_date,end_date,ind).to_pickle(save_path + 'Factor_NorthMaxdivPositve' + '.pkl')
    Factor_12and1Momentum(start_date, end_date, ind, period='M').to_pickle(save_path + 'Factor_12and1Momentum' + '.pkl')






