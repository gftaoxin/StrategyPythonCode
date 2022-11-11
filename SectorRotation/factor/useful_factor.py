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
################################################## 量价因子（7） #################################################
def Factor_DragonDifference(start_date,end_date,ind):
    # 逻辑：将板块内的个股拆分为大市值个股和小市值个股；关注版块内大市值个股的表现，和板块内整个个股走势的表现。
    # 在相同的市场行情下，分化度越小的小的板块，表明行情的同步性越强，后面表现会更为一致；分化度越大的板块，表明行情的同步性较弱，豁免会出现回落。
    # （但是，同步性最高的板块，通常是无效板块（因为只有什么股票都没有变化，同步性才会最高，因此这部分板块属于不可预测的，要调整到中间位置）
    # 第一步都是一致的，先获取日期，行业名称，个股所属行业
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind,date_list)

    # 个股所属行业
    mv = get_daily_1factor('mkt_free_cap', date_list=date_list)
    mv_20days = mv.rolling(20).mean()
    close_badj = get_daily_1factor('close_badj', date_list=date_list)
    pct_20days = close_badj.pct_change(20,fill_method=None)[~np.isnan(close_badj)]

    dragon_factor = pd.DataFrame(index=date_list,columns=ind_name)
    for i in ind_name:
        dragon_pct = pct_20days[mv_20days[code_ind == i].rank(pct=True,axis=1) >=0.4].mean(axis=1)
        other_pct = pct_20days[code_ind == i].mean(axis=1)
        dragon_factor[i] = (dragon_pct - other_pct)

    dragon_factor = dragon_factor[ind_useful].dropna(how='all',axis=1).dropna(how='all')
    factor =  (dragon_factor.sub(dragon_factor.mean(axis=1),axis=0).div(dragon_factor.std(axis=1),axis=0)) ** 2
    # 对于排名低的10%个行业，通常和市场是同步的，那么向上向下是不可预估的，所以把该部分变成中位数
    middle = pd.concat([factor.median(axis=1).rename(x) for x in factor.columns], axis=1)

    factor = pd.DataFrame(np.where((factor.rank(axis=1).T <= round(ind_useful.loc[factor.index].sum(axis=1)/10)).T,
                                   middle, factor),index=factor.index, columns=factor.columns)

    factor = pd.DataFrame(np.where((factor.rank(axis=1,pct=True) <= 0.1),
                                   middle, factor),index=factor.index, columns=factor.columns)

    # 给结果根据行业的情况赋值
    factor = -factor.loc[get_pre_trade_date(start_date, offset=1):end_date].dropna(how='all')

    return factor

def Factor_OverNightMomentum(start_date,end_date,ind):
    # 逻辑：隔夜溢价率，个股最近20日的隔夜溢价率，表现出明显的反转效应。
    # 通常表现为，机构资金通常不会在隔夜大幅下单，如果隔夜的溢价率较高，表明是散户的行为，那么过于一个月散户行为较为明显的，未来收益会较弱
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)

    ind_close =get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_open = get_daily_1factor('open', date_list=date_list, code_list=ind_name, type=ind[:-1])

    overnight_factor = (ind_open / ind_close.shift(1)-1).rolling(20).apply(lambda x: (1+x).prod()-1)

    # 给结果根据行业的情况赋值
    overnight_factor = overnight_factor[ind_useful].loc[get_pre_trade_date(start_date,offset=1):end_date]

    return -overnight_factor

def Factor_HighMomentum(start_date,end_date,ind):
    # 逻辑：隔夜溢价率，个股最近20日的隔夜溢价率，表现出明显的反转效应。
    # 通常表现为，机构资金通常不会在隔夜大幅下单，如果隔夜的溢价率较高，表明是散户的行为，那么过于一个月散户行为较为明显的，未来收益会较弱
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)

    ind_close =get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_high = get_daily_1factor('high', date_list=date_list, code_list=ind_name, type=ind[:-1])

    overnight_factor = (ind_high / ind_close.shift(1)-1).rolling(20).apply(lambda x: (1+x).prod()-1)
    overnight_factor = overnight_factor[ind_useful].dropna(how='all').dropna(how='all', axis=1)
    # 给结果根据行业的情况赋值
    factor = (overnight_factor.sub(overnight_factor.mean(axis=1), axis=0).div(overnight_factor.std(axis=1), axis=0)) ** 2

    return -factor.loc[get_pre_trade_date(start_date,offset=1):end_date]

def Factor_bigsmall_order(start_date,end_date,ind,period='M'):

    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date, period=period)
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).dropna(how='all')
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]
    ind_useful = get_useful_ind(ind, date_list)
    # 大单指标
    big_order_buy = get_moneyflow_data('BUY_VALUE_EXLARGE_ORDER',date_list=date_list)
    big_order_sell = get_moneyflow_data('SELL_VALUE_EXLARGE_ORDER',date_list=date_list)
    small_order_buy = get_moneyflow_data('BUY_VALUE_SMALL_ORDER', date_list=date_list)
    small_order_sell = get_moneyflow_data('SELL_VALUE_SMALL_ORDER', date_list=date_list)
    # 自由流通市值
    mv = (get_daily_1factor('mkt_free_cap', date_list=date_list))

    daily_big_in = pd.DataFrame(index=date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (big_order_buy -big_order_sell - ((small_order_buy - small_order_sell)))[code_ind == i].sum(axis=1) / mv[code_ind == i].sum(axis=1)
        daily_big_in[i] = factor_i

    factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in range(1, len(period_date_list)):
        date, last_date = period_date_list[i], period_date_list[i - 1]
        factor.loc[date] = daily_big_in.loc[last_date:date].sum()
    factor = factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)
    return factor

def Factor_bigsmall_amt_order(start_date, end_date, ind, period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date, period=period)
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).dropna(how='all')
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]
    ind_useful = get_useful_ind(ind, date_list)
    # 大单指标
    big_order_buy = get_moneyflow_data('BUY_VALUE_EXLARGE_ORDER', date_list=date_list)
    big_order_sell = get_moneyflow_data('SELL_VALUE_EXLARGE_ORDER', date_list=date_list)
    small_order_buy = get_moneyflow_data('BUY_VALUE_SMALL_ORDER', date_list=date_list)
    small_order_sell = get_moneyflow_data('SELL_VALUE_SMALL_ORDER', date_list=date_list)
    # 自由流通市值
    amt = (get_daily_1factor('amt', date_list=date_list))

    daily_big_in = pd.DataFrame(index=date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (big_order_buy - big_order_sell - ((small_order_buy - small_order_sell)))[code_ind == i].sum(axis=1) / amt[code_ind == i].sum(axis=1)
        daily_big_in[i] = factor_i

    factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in range(1, len(period_date_list)):
        date, last_date = period_date_list[i], period_date_list[i - 1]
        factor.loc[date] = daily_big_in.loc[last_date:date].sum()

    factor = factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)

    return factor

def Factor_sharpe(start_date,end_date,ind,period='M'):
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date, period=period)
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    # 个股所属行业
    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_pct = ind_close.pct_change(1,fill_method=None)
    factor = pd.DataFrame(index=period_date_list, columns=ind_pct.columns)
    for i in range(1, len(period_date_list)):
        date, last_date = period_date_list[i], period_date_list[i - 1]
        pct_month = ind_close.loc[date] / ind_close.loc[last_date] - 1

        factor.loc[date] = pct_month / ind_pct.loc[last_date:date].std()

    factor = factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)


    return factor

def Factor_jensen(start_date,end_date,ind,period='M'):
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date, period=period)
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    # 个股所属行业
    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_pct = ind_close.pct_change(1,fill_method=None)
    bench_close = get_daily_1factor('close',date_list=date_list,type='bench')['wind_A']
    bench_pct = bench_close.pct_change(1,fill_method=None)

    factor = pd.DataFrame(index=period_date_list, columns=ind_pct.columns)
    for i in range(1, len(period_date_list)):
        date, last_date = period_date_list[i], period_date_list[i - 1]
        beta = pd.Series(index = ind_name)
        for t in ind_name:
            beta.loc[t] = np.cov(ind_pct.loc[last_date:date,t], bench_pct.loc[last_date:date])[0,1] / bench_pct.loc[last_date:date].std()

        pct_month = ind_close.loc[date] / ind_close.loc[last_date] - 1
        bench_pct_month = bench_pct.loc[date] / bench_pct.loc[last_date] - 1

        factor.loc[date] = pct_month - beta * bench_pct_month

    factor = factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)


    return factor

##################################################### 北向资金因子（7） ##################################################
def north_volume_net_in(date_list):
    north_volume = get_daily_1factor('north_quantity', date_list=date_list).dropna(how='all')
    north_volume_in = north_volume.diff(1).dropna(how='all')
    # 获取分红除息数据
    code_div = getEXRightDividend()
    code_div = code_div[code_div['date'].isin(north_volume_in.index[2:])][code_div['code'].isin(north_volume_in.columns)]
    code_div = code_div[code_div['shareRatio'] > 0]

    for index in code_div.index:
        date,code = code_div.loc[index,'date'],code_div.loc[index,'code']
        shareRatio = code_div.loc[index,'shareRatio']

        change_date = get_pre_trade_date(date)
        if north_volume.loc[get_pre_trade_date(change_date), code] >0:
            north_volume_in.loc[change_date,code] = round(north_volume.loc[change_date,code] - north_volume.loc[get_pre_trade_date(change_date), code] * (1 +shareRatio))

    return north_volume_in
def Factor_NorthAmtIn(start_date,end_date,ind):
    # 逻辑：在个股成交额中，北向资金占总成交额的比例。在此处并不区分方向是流入还是流出，只区分成交量
    # 也就是说，如果北向资金在过去的成交量较大，说明北向资金关注该板块，过去20日内北向资金关注的板块会有更多的超额
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())

    ind_useful = get_useful_ind(ind, date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]

    close = get_daily_1factor('vwap',date_list=date_list).dropna(how='all')
    north_amt = (abs(north_volume_in) * close).dropna(how='all')[stock_pool]
    amt = get_daily_1factor('amt', date_list=date_list)[stock_pool]


    north_ind_amt = pd.concat([amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name],axis=1).dropna(how='all')
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name],axis=1)

    factor = (north_ind_amt_in[ind_useful] / (north_ind_amt[ind_useful] * 1000)).dropna(how='all').dropna(how='all',axis=1)
    factor = factor.rolling(20).sum()

    factor = factor[ind_useful].dropna(how='all')


    return factor.loc[get_pre_trade_date(start_date,offset=1):end_date]

def Factor_NorthPositiveWeight(start_date,end_date,ind):
    # 逻辑：北向资金在过去120日内流入的金额（不关心流出），占据行业总自由流通市值的比例。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]

    vwap = get_daily_1factor('vwap', date_list=date_list).dropna(how='all')[stock_pool]
    north_amt = (north_volume_in * vwap).dropna(how='all')
    free_mv = get_daily_1factor('mkt_free_cap', date_list=date_list)

    north_ind_mv = pd.concat([free_mv[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)[ind_useful].dropna(how='all')
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)

    north_ind_positive_in = north_ind_amt_in.copy()
    north_ind_positive_in[north_ind_positive_in < 0 ] = 0

    factor = north_ind_positive_in.rolling(120).sum() / north_ind_mv
    # 给结果根据行业的情况赋值
    factor = factor[ind_useful].dropna(how='all')

    return factor.loc[get_pre_trade_date(start_date,offset=1):end_date]

def Factor_NorthNegitiveWeight(start_date,end_date,ind):
    # 逻辑：北向资金在过去120日内流入的金额（不关心流出），占据行业总自由流通市值的比例。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]

    vwap = get_daily_1factor('vwap', date_list=date_list).dropna(how='all')[stock_pool]
    north_amt = (north_volume_in * vwap).dropna(how='all')
    free_mv = get_daily_1factor('mkt_free_cap', date_list=date_list)

    north_ind_mv = pd.concat([free_mv[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)[ind_useful].dropna(how='all')
    north_ind_amt_in = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name], axis=1)

    north_ind_positive_out = north_ind_amt_in.copy()
    north_ind_positive_out[north_ind_positive_out > 0] = 0

    factor = north_ind_positive_out.rolling(120).sum() / north_ind_mv
    factor = factor[ind_useful].dropna(how='all')

    return -factor.loc[get_pre_trade_date(start_date,offset=1):end_date]

def Factor_NorthPositiveRate(start_date,end_date,ind):
    # 逻辑：北向资金在过去120日净流入情况 / 北向资金在过去120日的流入情况。
    # 即，如果这个比例越高，那么北向资金在过去120日绝大多数情况下都是流入的。 但北向资金通常有明显的持续性和反转，大幅度净流入和大幅度净流出均有效。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=150), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]

    close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
    north_amt = (north_volume_in * close).dropna(how='all')

    north_ind_amt_in = pd.DataFrame(index=north_amt.index, columns=ind_name)
    north_ind_positive_in = pd.DataFrame(index=north_amt.index, columns=ind_name)
    for i in ind_name:
        for x in north_ind_amt_in.index:
            history_amt = north_amt.loc[get_pre_trade_date(x, offset=119):x,
                          code_ind.loc[x][code_ind.loc[x] == i].index].sum(axis=1)
            north_ind_amt_in.loc[x, i] = history_amt.sum()
            north_ind_positive_in.loc[x, i] = history_amt[history_amt > 0].sum()

    factor = (north_ind_amt_in / north_ind_positive_in)[ind_useful]
    factor = factor.replace([np.inf, -np.inf], np.nan)[ind_useful].dropna(how='all')
    # 给结果根据行业的情况赋值
    factor = factor[ind_useful].dropna(how='all') ** 2

    return factor.loc[max(get_pre_trade_date(start_date, offset=1),20161201):end_date].astype(float)

def Factor_NorthPositiveRate_Unline(start_date,end_date,ind):
    # 逻辑：北向资金在过去120日净流入情况 / 北向资金在过去120日的流入情况。
    # 进行了如下变换：①先对于因子进行平方，即净流出占比高或者净流出占比高的方向，都比较高
    # ②再对因子归一化，归一化后再平方，平方后进行非线性回归，取中间。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=150), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]

    close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
    north_amt = (north_volume_in * close).dropna(how='all')

    north_ind_amt_in = pd.DataFrame(index=north_amt.index,columns=ind_name)
    north_ind_positive_in = pd.DataFrame(index=north_amt.index,columns=ind_name)
    for i in ind_name:
        for x in north_ind_amt_in.index:
            history_amt = north_amt.loc[get_pre_trade_date(x,offset=119):x,code_ind.loc[x][code_ind.loc[x] == i].index].sum(axis=1)
            north_ind_amt_in.loc[x,i] = history_amt.sum()
            north_ind_positive_in.loc[x,i] =history_amt[history_amt >0].sum()

    factor = (north_ind_amt_in / north_ind_positive_in)[ind_useful]
    factor = factor.replace([np.inf,-np.inf],np.nan)[ind_useful].dropna(how='all')
    # 给结果根据行业的情况赋值
    factor = (deal_data(factor** 2) ** 2).dropna(how='all')

    new_factor = pd.DataFrame(index=factor.index, columns=factor.columns)
    for i in new_factor.index:
        new_factor.loc[i] = get_regression(factor.loc[i].dropna(), (factor**3).loc[i].dropna(), type='residual')

    return new_factor.loc[max(get_pre_trade_date(start_date, offset=1),20161201):end_date].astype(float)

def Factor_NorthWeight(start_date,end_date,ind):
    # 逻辑：北向资金过去20日的净流入 / 北向资金过去20日在所有行业的净流入 （ 即北向资金净流入有多少比例流入到了该行业中）
    # 但是因为存在市值的影响，所以先根据流入比例除以市值占比，观测真实流入情况。
    # 同样，流入最多的和流出最多的在未来面临动量和反转的机会
    date_list = get_date_range(get_pre_trade_date(start_date, offset=150), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    code_ind = get_daily_1factor(ind, date_list=date_list)[stock_pool]

    close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
    north_amt = (north_volume_in * close).dropna(how='all')

    ind_mv = get_modified_ind_mv(date_list=date_list, ind_type=ind)

    north_ind_amt_in = pd.DataFrame(index=north_amt.index, columns=ind_name)
    for i in ind_name:
        for x in north_ind_amt_in.index:
            history_amt = north_amt.loc[get_pre_trade_date(x, offset=59):x,code_ind.loc[x][code_ind.loc[x] == i].index].sum(axis=1)
            north_ind_amt_in.loc[x, i] = history_amt.sum()

    factor = (north_ind_amt_in.T / north_ind_amt_in.sum(axis=1)).T

    factor = factor / (ind_mv.div(ind_mv.sum(axis=1),axis=0))

    factor =factor[ind_useful] ** 2

    return factor.loc[max(get_pre_trade_date(start_date, offset=1),20161201):end_date].astype(float)

def Factor_north_bigin(start_date,end_date,ind,period='M'):
    # 逻辑：北向资金本月流入金额/上个月原本持有某一只个股的市值，表明北向资金是否相对于自身大幅度加仓
    date_list = get_date_range(get_pre_trade_date(start_date, offset=150), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    code_ind = get_daily_1factor(ind, date_list=date_list)
    north_volume_in = north_volume_net_in(date_list)

    stock_pool = clean_stock_list(start_date=date_list[0], end_date=date_list[-1])

    close = get_daily_1factor('vwap', date_list=date_list).dropna(how='all')
    north_volume = get_daily_1factor('north_quantity', date_list=date_list).dropna(how='all')
    north_amt_in = north_volume_in * close

    period_holding_date = get_date_range(date_list[0],date_list[-1],period=period)
    north_mv_20days =  pd.DataFrame(index=period_holding_date, columns=north_volume.columns)
    north_amt_in20days = pd.DataFrame(index=period_holding_date, columns=north_volume.columns)
    for i in range(1,len(period_holding_date)):
        date,last_date = period_holding_date[i],period_holding_date[i-1]
        north_mv_20days.loc[date] = north_volume.loc[last_date:date].mean()
        north_amt_in20days.loc[date] = north_amt_in.loc[last_date:date].sum()

    flowratio = (north_amt_in20days / north_mv_20days).dropna(how='all')[stock_pool]
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[flowratio.index][stock_pool]

    ind_flowratio = pd.DataFrame(index=flowratio.index, columns=ind_name)
    for i in ind_name:
        ind_flowratio[i] = (flowratio[code_ind == i] *  mv[code_ind == i].div(mv[code_ind == i].sum(axis=1),axis=0)).sum(axis=1)
        #ind_flowratio[i] = flowratio[code_ind == i].mean(axis=1)

    factor = ind_flowratio[ind_useful].dropna(how='all').astype(float).loc[get_pre_trade_date(start_date, offset=1):end_date]

    #factor = ind_flowratio.rolling(20).mean()[ind_useful].dropna(how='all')
    #factor = (deal_data(factor) ** 2)


    return factor

##################################################### 取残差因子（4) ##################################################
def Factor_ResidualMometumn_2barra_sizebtop(start_date,end_date,ind,period='M'):
    # 逻辑：残差动量，即单纯的动量因子可能受到该区间内某些表现较好的风险因子的影响；因此风险因子发生反转时，那么动量因子就容易失效。
    # 因此，先通过回归剥离风险因子的影响，得到残差，规避风险因子对策略的影响。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=520), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=520), end_date, period=period)

    size = deal_data(Barra_size(get_pre_trade_date(start_date, offset=520), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)[ind_useful]
    btop = deal_data(Barra_BTOP(get_pre_trade_date(start_date, offset=520), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)[ind_useful]

    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1]).loc[period_date_list].dropna(how='all')
    pct_20days = ind_close.pct_change(fill_method=None)
    # 开始进行时序回归
    factor1 = pd.DataFrame(index=period_date_list,columns=ind_name)
    factor2= pd.DataFrame(index=period_date_list, columns=ind_name)
    factor3 = pd.DataFrame(index=period_date_list, columns=ind_name)

    for i in ind_name:
        df_y = pct_20days[[i]].dropna(how='all').astype(float)
        df_x = pd.concat([size[i].rename('size').dropna(),btop[i].rename('btop').dropna()],axis=1).dropna().astype(float)
        common_date = sorted(list(set(df_y.index).intersection(df_x.index)))
        regression_residual1 = ts_rolling_regression(df_x.loc[common_date].shift(1).dropna(how='all'),
                                                    df_y.loc[common_date].shift(1).dropna(how='all'),
                                                    rolling_days=12)[1]
        regression_residual2 = ts_rolling_regression(df_x.loc[common_date].shift(1).dropna(how='all'),
                                                    df_y.loc[common_date].shift(1).dropna(how='all'),
                                                    rolling_days=6)[1]
        regression_residual3 = ts_rolling_regression(df_x.loc[common_date].shift(1).dropna(how='all'),
                                                    df_y.loc[common_date].shift(1).dropna(how='all'),
                                                    rolling_days=24)[1]

        factor1[i] = regression_residual1[i]
        factor2[i] = regression_residual2[i]
        factor3[i] = regression_residual3[i]

    factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0))[ind_useful]
    factor = -factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)

    return factor

def Factor_ResidualOverNightMometumn_2barra_sizebtop(start_date,end_date,ind,period='M'):
    # 逻辑：残差动量，即单纯的动量因子可能受到该区间内某些表现较好的风险因子的影响；因此风险因子发生反转时，那么动量因子就容易失效。
    # 因此，先通过回归剥离风险因子的影响，得到残差，规避风险因子对策略的影响。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=520), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=520), end_date, period=period)

    size = deal_data(Barra_size(get_pre_trade_date(start_date, offset=520), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)[ind_useful]
    btop = deal_data(Barra_BTOP(get_pre_trade_date(start_date, offset=520), end_date, ind).reindex(period_date_list).astype(float).dropna(how='all')).fillna(0)[ind_useful]

    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1]).dropna(how='all')
    ind_open = get_daily_1factor('open', date_list=date_list, code_list=ind_name, type=ind[:-1]).dropna(how='all')
    overnight_factor = (ind_open / ind_close.shift(1) - 1).rolling(20).apply(lambda x: (1 + x).prod() - 1)

    # 开始进行时序回归
    factor1 = pd.DataFrame(index=period_date_list,columns=ind_name)
    factor2= pd.DataFrame(index=period_date_list, columns=ind_name)

    for i in ind_name:
        df_y = overnight_factor[[i]].dropna(how='all').astype(float)
        df_x = pd.concat([size[i].rename('size').dropna(),btop[i].rename('btop').dropna()],axis=1).dropna().astype(float)
        common_date = sorted(list(set(df_y.index).intersection(df_x.index)))
        regression_residual1 = ts_rolling_regression(df_x.loc[common_date].shift(1).dropna(how='all'),
                                                    df_y.loc[common_date].shift(1).dropna(how='all'),
                                                    rolling_days=12)[1]
        regression_residual2 = ts_rolling_regression(df_x.loc[common_date].shift(1).dropna(how='all'),
                                                    df_y.loc[common_date].shift(1).dropna(how='all'),
                                                    rolling_days=6)[1]

        factor1[i] = regression_residual1[i]
        factor2[i] = regression_residual2[i]


    factor = (factor1.fillna(0) + factor2.fillna(0))[ind_useful]
    factor = -factor.loc[get_pre_trade_date(start_date, offset=1):end_date][ind_useful].astype(float)

    return factor

def Factor_residual_momentum(start_date,end_date,ind,period='M'):
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date, period=period)
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    #stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date, offset=120), end_date=end_date)
    # 个股所属行业
    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_amt = get_daily_1factor('amt', date_list=date_list, code_list=ind_name, type=ind[:-1])
    amt_ratio = ind_amt/ ind_amt.shift(1) - 1
    ind_pct = ind_close / ind_close.shift(1) - 1

    factor = pd.DataFrame(index=period_date_list, columns=ind_pct.columns)
    for i in range(12, len(period_date_list)):
        date, last_date = period_date_list[i], period_date_list[i - 12]
        for t in ind_name:
            df_y = ind_pct.loc[last_date:date,t].dropna()
            df_x = amt_ratio.loc[last_date:date,t].dropna()
            if len(df_x) >0:
                wls = sm.WLS(df_y,df_x).fit()
                factor.loc[date,t] = wls.resid.sum() / df_y.std()

    factor = factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)

    return factor

def Factor_residual_overnight(start_date,end_date,ind,period='M'):
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date, period=period)
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    ind_useful = get_useful_ind(ind, date_list)

    ind_close = get_daily_1factor('close',date_list=date_list,code_list=ind_name,type=ind[:-1])
    ind_open = get_daily_1factor('open', date_list=date_list, code_list=ind_name, type=ind[:-1])
    ind_amt = get_daily_1factor('amt', date_list=date_list, code_list=ind_name, type=ind[:-1])
    amt_ratio = ind_amt/ ind_amt.rolling(10).mean() - 1
    ind_pct = ind_open / ind_close.shift(1) - 1

    factor = pd.DataFrame(index=period_date_list, columns=ind_pct.columns)
    for i in range(12, len(period_date_list)):
        date, last_date = period_date_list[i], period_date_list[i - 12]
        for t in ind_name:
            df_y = ind_pct.loc[last_date:date,t].dropna()
            df_x = amt_ratio.loc[last_date:date,t].dropna()
            all_index = (set(df_y.index)).intersection(df_x.index)
            if len(df_x) >0:
                wls = sm.WLS(df_y.loc[all_index],df_x.loc[all_index]).fit()
                factor.loc[date,t] = wls.resid.sum() / df_y.std()

    factor = factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date].astype(float)

    return factor

#################################################### 财务因子（9） #######################################################
def Factor_ROA(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    net_asset = get_quarter_1factor('TOT_ASSETS', 'AShareBalanceSheet', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)
    net_income = get_quarter_1factor('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)

    asset_turnover = (net_income) / net_asset

    save_index = asset_turnover.index
    asset_turnover.index = pd.Series(asset_turnover.index).apply(lambda x: get_recent_trade_date(x))
    asset_turnover = asset_turnover[stock_pool]
    asset_turnover.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        asset_turnover.iloc[i::4] = asset_turnover.iloc[i::4].ffill(limit=2)

    yoy_diff = get_yoy(asset_turnover)#.diff(1)
    diff_expect = yoy_diff.rolling(8).mean()
    diff_std = yoy_diff.rolling(8).std()

    new_factor = (yoy_diff - diff_expect) / diff_std
    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]
    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_SUE_diff(start_date,end_date,ind,period='M'):
    # 逻辑：残差动量，即单纯的动量因子可能受到该区间内某些表现较好的风险因子的影响；因此风险因子发生反转时，那么动量因子就容易失效。
    # 因此，先通过回归剥离风险因子的影响，得到残差，规避风险因子对策略的影响。
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    earning = get_quarter_1factor('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type='408001000',
                                  date_list=date_list).dropna(how='all', axis=1)
    save_index = earning.index
    earning.index = pd.Series(earning.index).apply(lambda x: get_recent_trade_date(x))
    earning = earning[stock_pool]
    earning.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        earning.iloc[i::4] = earning.iloc[i::4].ffill(limit=2)

    earnings_yoy_diff = get_yoy(earning).diff(1)
    expect_earning = earnings_yoy_diff.rolling(8).mean()
    std_earning = earnings_yoy_diff.rolling(8).std()

    SUE = (earnings_yoy_diff - expect_earning) / std_earning
    SUE = fill_quarter2daily_by_issue_date(SUE, 'AShareIncome', '408001000', keep=None).reindex(
        period_date_list).dropna(how='all')
    SUE = SUE[stock_pool]
    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[SUE.index]

    ind_SUE = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        SUE_i = (SUE[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_SUE[i] = SUE_i

    factor = ind_SUE[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_asset_turnover_diff(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    asset = get_quarter_1factor('TOT_ASSETS', 'AShareBalanceSheet', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)
    operate_income = get_quarter_1factor('TOT_OPER_REV', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)

    asset_turnover = operate_income / asset

    save_index = asset_turnover.index
    asset_turnover.index = pd.Series(asset_turnover.index).apply(lambda x: get_recent_trade_date(x))
    asset_turnover = asset_turnover[stock_pool]
    asset_turnover.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        asset_turnover.iloc[i::4] = asset_turnover.iloc[i::4].ffill(limit=2)


    yoy_diff = get_yoy(asset_turnover).diff()
    diff_expect = yoy_diff.rolling(8).mean()
    diff_std = yoy_diff.rolling(8).std()

    new_factor = (yoy_diff - diff_expect) / diff_std
    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]
    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_GPM(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    revnue = get_quarter_1factor('OPER_REV', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)
    cost = get_quarter_1factor('LESS_OPER_COST', 'AShareIncome', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)
    #oper_profit = get_quarter_1factor('TOT_ASSETS', 'AShareBalanceSheet', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)

    asset_turnover = (revnue - cost) / revnue

    save_index = asset_turnover.index
    asset_turnover.index = pd.Series(asset_turnover.index).apply(lambda x: get_recent_trade_date(x))
    asset_turnover = asset_turnover[stock_pool]
    asset_turnover.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        asset_turnover.iloc[i::4] = asset_turnover.iloc[i::4].ffill(limit=2)

    yoy_diff = get_yoy(asset_turnover)#.diff(1)
    diff_expect = yoy_diff.rolling(8).mean().astype(float).round(5)
    diff_std = yoy_diff.rolling(8).std().astype(float).round(5)

    new_factor = (yoy_diff - diff_expect) / diff_std
    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]
    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_GPM_meanstd(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    revnue = get_quarter_1factor('OPER_REV', 'AShareIncome', report_type='408001000', date_list=date_list).dropna(
        how='all', axis=1)
    cost = get_quarter_1factor('LESS_OPER_COST', 'AShareIncome', report_type='408001000', date_list=date_list).dropna(
        how='all', axis=1)

    report_data = (revnue - cost) / revnue

    save_index = report_data.index
    report_data.index = pd.Series(report_data.index).apply(lambda x: get_recent_trade_date(x))
    report_data = report_data[stock_pool]
    report_data.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        report_data.iloc[i::4] = report_data.iloc[i::4].ffill(limit=2)

    new_factor = report_data.replace(0, np.nan).diff().round(5) / abs(report_data.replace(0, np.nan).shift(1))
    new_factor = new_factor.T.apply(lambda x:mad(x)).T

    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]

    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_AT(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    revnue = get_quarter_1factor('OPER_REV', 'AShareIncome', report_type='408001000', date_list=date_list).dropna(
        how='all', axis=1)
    asset = get_quarter_1factor('TOT_ASSETS', 'AShareBalanceSheet', report_type='408001000', date_list=date_list).dropna(
        how='all', axis=1)

    report_data = revnue.round(5) / asset.replace(0,np.nan)

    save_index = report_data.index
    report_data.index = pd.Series(report_data.index).apply(lambda x: get_recent_trade_date(x))
    report_data = report_data[stock_pool]
    report_data.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        report_data.iloc[i::4] = report_data.iloc[i::4].ffill(limit=2)

    new_factor = report_data.T.apply(lambda x:mad(x)).T
    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]

    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_APR_diff(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    net_profit = get_quarter_1factor('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)
    cash_flow = get_quarter_1factor('NET_CASH_FLOWS_OPER_ACT', 'AShareCashFlow', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)

    opr_revnue = get_quarter_1factor('TOT_OPER_REV', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)
    opr_cost = get_quarter_1factor('TOT_OPER_COST', 'AShareIncome', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)

    report_data = (net_profit - cash_flow).round(5) / (opr_revnue - opr_cost).replace(0,np.nan)

    save_index = report_data.index
    report_data.index = pd.Series(report_data.index).apply(lambda x: get_recent_trade_date(x))
    report_data = report_data[stock_pool]
    report_data.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        report_data.iloc[i::4] = report_data.iloc[i::4].ffill(limit=2)


    new_factor = report_data.diff()
    new_factor = new_factor.T.apply(lambda x:mad(x)).T

    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]

    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

def Factor_QR_diff2(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(
        period_date_list).dropna(how='all')

    curr_asset = get_quarter_1factor('TOT_CUR_ASSETS', 'AShareBalanceSheet', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)
    inventory = get_quarter_1factor('INVENTORIES', 'AShareBalanceSheet', report_type='408001000', date_list=date_list).dropna(how='all', axis=1)
    prepay = get_quarter_1factor('PREPAY', 'AShareBalanceSheet', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)
    curr_liab = get_quarter_1factor('TOT_CUR_LIAB', 'AShareBalanceSheet', report_type='408001000',date_list=date_list).dropna(how='all', axis=1)

    inventory = inventory.reindex(columns=curr_asset.columns).replace(np.nan,0)
    prepay = prepay.reindex(columns=curr_asset.columns).replace(np.nan,0)

    report_data = (curr_asset - inventory - prepay).round(5) / curr_liab.replace(0,np.nan)

    save_index = report_data.index
    report_data.index = pd.Series(report_data.index).apply(lambda x: get_recent_trade_date(x))
    report_data = report_data[stock_pool]
    report_data.index = save_index
    # 存在一些异常值，进行填充，但因为是最新财报，所以间隔4个财报季度填充
    for i in [0, 1, 2, 3]:
        report_data.iloc[i::4] = report_data.iloc[i::4].ffill(limit=2)

    new_factor = report_data.diff()
    new_factor = new_factor.T.apply(lambda x:mad(x)).T

    new_factor = fill_quarter2daily_by_fixed_date(new_factor, keep='last').reindex(period_date_list).dropna(how='all')
    new_factor = new_factor[stock_pool]

    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = deal_data(ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date])  ** 2

    return factor

def Factor_epredict_eps_3m(start_date,end_date,ind,period='M'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=1200), end_date)
    ind_name = list(get_real_ind(ind_type=ind[:-1], level=int(ind[-1])).keys())
    code_ind = get_daily_1factor(ind, date_list=date_list)
    ind_useful = get_useful_ind(ind, date_list)
    period_date_list = get_date_range(max(get_pre_trade_date(start_date, offset=1200),20100101), end_date, period=period)
    # 净利润
    stock_pool = clean_stock_list(least_live_days=1, no_ST=False, no_pause=False, no_limit_up=False,
                                  start_date=date_list[0], end_date=period_date_list[-1]).reindex(period_date_list).dropna(how='all')

    table = 'ConsensusExpectationFactor'
    factor_str = 'S_INFO_WINDCODE,TRADE_DT, S_WEST_NETPROFIT_FTM_CHG_1M,S_WEST_NETPROFIT_FTM_CHG_3M,S_WEST_NETPROFIT_FTM_CHG_6M, S_WEST_EPS_FTM_CHG_1M, S_WEST_EPS_FTM_CHG_3M, S_WEST_EPS_FTM_CHG_6M'

    eps_3m = pd.DataFrame()
    for date in period_date_list:
        sql = r"select %s from wind.%s a where a.TRADE_DT = '%s'" % (factor_str, table, date)
        data_values = pd.read_sql(sql, con)
        eps_3m = pd.concat([eps_3m, data_values.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_WEST_EPS_FTM_CHG_3M')])

    new_factor = eps_3m.copy()
    new_factor.index = new_factor.index.astype(int)
    new_factor.columns = pd.Series(new_factor.columns).apply(lambda x:trans_windcode2int(x))
    new_factor = new_factor.T.apply(lambda x:mad(x)).T
    new_factor = new_factor[stock_pool]

    # 按行业累加起来
    mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list)).loc[new_factor.index]

    ind_factor = pd.DataFrame(index=period_date_list, columns=ind_name)
    for i in ind_name:
        factor_i = (new_factor[code_ind == i] * (mv[code_ind == i]).div(mv[code_ind == i].sum(axis=1), axis=0)).sum(axis=1)
        ind_factor[i] = factor_i

    factor = ind_factor[ind_useful].loc[get_pre_trade_date(start_date, offset=1):end_date]

    return factor

if __name__ == '__main__':
    start_date,end_date = 20131201,20221102
    north_start_date = 20160601
    ind = 'SW1'
    save_path = 'E:/FactorTest/useful_factor/'

    Factor_DragonDifference(start_date, end_date, ind).to_pickle(save_path + 'Factor_DragonDifference' + '.pkl')
    Factor_OverNightMomentum(start_date,end_date,ind).to_pickle(save_path + 'Factor_OverNightMomentum' + '.pkl')
    Factor_HighMomentum(start_date,end_date,ind).to_pickle(save_path + 'Factor_HighMomentum' + '.pkl')
    Factor_bigsmall_order(start_date,end_date,ind,'M').to_pickle(save_path + 'Factor_bigsmall_order' + '.pkl')
    Factor_bigsmall_amt_order(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_bigsmall_amt_order' + '.pkl')
    Factor_sharpe(start_date, end_date, ind, period='M').to_pickle(save_path + 'Factor_sharpe' + '.pkl')
    Factor_jensen(start_date, end_date, ind, period='M').to_pickle(save_path + 'Factor_jensen' + '.pkl')

    Factor_NorthAmtIn(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_NorthAmtIn' + '.pkl')
    Factor_NorthPositiveWeight(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_NorthPositiveWeight' + '.pkl')
    Factor_NorthNegitiveWeight(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_NorthNegitiveWeight' + '.pkl')
    Factor_NorthPositiveRate(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_NorthPositiveRate' + '.pkl')
    Factor_NorthPositiveRate_Unline(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_NorthPositiveRate_Unline' + '.pkl')
    Factor_NorthWeight(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_NorthWeight' + '.pkl')
    Factor_north_bigin(north_start_date, end_date, ind).to_pickle(save_path + 'Factor_north_bigin' + '.pkl')

    Factor_ResidualMometumn_2barra_sizebtop(start_date, end_date, ind).to_pickle(save_path + 'Factor_ResidualMometumn_2barra_sizebtop' + '.pkl')
    Factor_ResidualOverNightMometumn_2barra_sizebtop(start_date, end_date, ind).to_pickle(save_path + 'Factor_ResidualOverNightMometumn_2barra_sizebtop' + '.pkl')
    Factor_residual_momentum(start_date, end_date, ind, period='M').to_pickle(save_path + 'Factor_residual_momentum' + '.pkl')
    Factor_residual_overnight(start_date,end_date,ind,period='M').to_pickle(save_path + 'Factor_residual_overnight' + '.pkl')

    Factor_ROA(start_date, end_date, ind,'M').to_pickle(save_path + 'Factor_SUE' + '.pkl')
    Factor_SUE_diff(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_SUE_diff' + '.pkl')
    Factor_asset_turnover_diff(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_asset_turnover_diff' + '.pkl')
    Factor_GPM(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_GPM' + '.pkl')
    Factor_GPM_meanstd(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_GPM_meanstd' + '.pkl')
    Factor_AT(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_AT' + '.pkl')
    Factor_APR_diff(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_APR_diff' + '.pkl')
    Factor_QR_diff2(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_QR_diff2' + '.pkl')
    Factor_epredict_eps_3m(start_date, end_date, ind, 'M').to_pickle(save_path + 'Factor_epredict_eps_3m' + '.pkl')


'''
self = FactorTest(test_start_date=20151231, test_end_date=20221101, ind='SW1', day=20,fee=0.001)

save_path = 'E:/FactorTest/useful_factor/'
factor_list = [x[:-4] for x in os.listdir(save_path)]
factor_result = pd.DataFrame(index = factor_list, columns=['ic','rank_ic','ICIR','excess_return','excess_sharpe','ls_return','ls_sharpe'])
for factor_name in factor_list:
    factor = pd.read_pickle(save_path + factor_name + '.pkl')
    box_in, test_result0, value_result0, ic0, rank_ic0 = self.cal_factor_result(factor, save_path=None)
    factor_result.loc[factor_name] = test_result0.loc[['ic','rank_ic','ICIR','excess_return','excess_sharpe','ls_return','ls_sharpe'],'all']

factor_result.loc[['Factor_DragonDifference','Factor_OverNightMomentum','Factor_HighMomentum','Factor_bigsmall_order','Factor_bigsmall_amt_order','Factor_sharpe','Factor_jensen']].mean()
factor_result.loc[['Factor_NorthAmtIn','Factor_NorthPositiveWeight','Factor_NorthNegitiveWeight','Factor_NorthPositiveRate','Factor_NorthPositiveRate_Unline','Factor_NorthWeight','Factor_north_bigin']].mean()
factor_result.loc[['Factor_ResidualMometumn_2barra_sizebtop','Factor_ResidualOverNightMometumn_2barra_sizebtop','Factor_residual_momentum','Factor_residual_overnight']].mean()
factor_result.loc[['Factor_SUE','Factor_SUE_diff','Factor_asset_turnover_diff','Factor_GPM','Factor_GPM_meanstd','Factor_AT','Factor_APR_diff','Factor_QR_diff2','Factor_epredict_eps_3m']].mean()
'''