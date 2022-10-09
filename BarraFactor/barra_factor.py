from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import statsmodels.api as sm
import numpy as np
#(1)处理极端值
def mad(x):
    median = np.nanmedian(x)
    mad = np.nanmedian(abs(x-median))

    high = median + 3 *1.4826 * mad
    low = median - 3 *1.4826 * mad

    x = np.where(x > high, high,x)
    x = np.where(x < low, low, x)

    return x
#(2)标准化
def standard(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    new_df = df.sub(mean,axis=0).div(std,axis=0)

    return new_df
# 1、去极值，标准化处理
def deal_data(df):
    new_df = df.T.apply(lambda x:mad(x)).T
    new_df = standard(new_df)

    new_df.dropna(how='all',axis=0)

    return new_df
# 2、获取半衰期权重
def get_half_weight(N, half_time):
    neg = 0.5**(1/half_time)

    x = [(1/neg**i) for i in range(1,N+1)]
    return [i/sum(x) for i in x]
# 3、两列的时序滚动回归
def ts_rolling_regression(df_x,df_y,rolling_days,weight=None):
    regression_predict = pd.DataFrame(index=df_y.index, columns=df_y.columns)
    regression_residual = pd.DataFrame(index=df_y.index, columns=df_y.columns)
    if df_y.shape[1] > 1:
        regression_coef = pd.DataFrame(index=df_y.index, columns=df_y.columns)
    else:
        regression_coef = pd.DataFrame(index=df_y.index, columns=df_x.columns)
    regression_residual_std = pd.DataFrame(index=df_y.index, columns=df_y.columns)

    for i in range(int(rolling_days/2), len(df_y)+1):
        if weight == None : # 表示用OLS回归
            wls = sm.WLS(df_y.iloc[max(i-rolling_days,0):i], df_x.iloc[max(i-rolling_days,0):i]).fit()
        else:
            if i < rolling_days:
                new_weight = [x/sum(weight[:len(df_x.iloc[max(i - rolling_days, 0):i])]) for x in weight[:len(df_x.iloc[max(i - rolling_days, 0):i])]]
            else:
                new_weight = weight.copy()
            wls = sm.WLS(df_y.iloc[max(i-rolling_days,0):i], df_x.iloc[max(i-rolling_days,0):i],weights=new_weight).fit()

        regression_predict.iloc[i-1] = wls.predict()[-1]
        regression_residual.iloc[i-1] = wls.resid.iloc[-1]
        regression_coef.iloc[i-1] = wls.params
        regression_residual_std.iloc[i-1] = wls.resid.std()

    return regression_predict,regression_residual,regression_coef, regression_residual_std
# 4、两行的线性回归
def get_regression(df_x,df_y,weight=None,type='predict'):
    if weight == None:  # 表示用OLS回归
        wls = sm.WLS(df_y, df_x).fit()
    else:
        wls = sm.WLS(df_y, df_x,weights=weight).fit()
    if type == 'predict':
        return wls.predict(df_x)
    else:
        return wls.resid
# 5、个股数据变为行业数据
def trans_factor_to_ind(df,ind='SW1'):
    df = df.dropna(how='all')
    date_list = df.index.to_list()
    ind_code = get_daily_1factor(ind, date_list)
    ind_useful = get_useful_ind(ind,date_list)
    inde_name = list(get_real_ind(ind[:-1],level=int(ind[-1])).keys())

    return pd.concat([df[ind_code == i].sum(axis=1).rename(i) for i in inde_name],axis=1)[ind_useful]
# 6、获取行业市值
def get_modified_ind_mv(date_list=None, code_list=None, ind_type='SW1'):
    mv = get_daily_1factor('mkt_free_cap', date_list, code_list).dropna(how='all')
    date_list = mv.index.to_list()
    code_list = mv.columns.to_list()

    ind = get_daily_1factor(ind_type, date_list, code_list)
    ind_codes = list(get_real_ind(ind_type[:-1],level=int(ind_type[-1])).keys())
    ind_uesful = get_useful_ind(ind_type, date_list)

    ind_result = np.r_['0,3', tuple(ind == x for x in ind_codes)]
    ind_mv = np.einsum('ijk,jk -> ijk',ind_result,mv)

    modified_ind_mv = pd.DataFrame(np.nansum(ind_mv,axis=2),index=ind_codes,columns=date_list).T

    return np.log(modified_ind_mv[ind_uesful]).replace([np.inf,-np.inf], np.nan)
# 9、获取实时的申万或者中信行业
# (1)获取实时的申万和中信行业（即根据时间会进行调整）
def get_real_ind(ind_type='SW',level=1):
    # 输入：sw,sw2021,CITICS,获取申万和中信对应的指数代码，指数名称
    level_dict = {1: ['一级行业代码', '一级行业名称'],
                  2: ['二级行业代码', '二级行业名称'],
                  3: ['三级行业代码', '三级行业名称']}

    if ind_type == 'SW':
        ind_name1 = pd.read_excel(base_address + '行业分类.xlsx', sheet_name='SW2021')
        ind_name2 = pd.read_excel(base_address + '行业分类.xlsx', sheet_name='SW')
        level = int(level) if type(level) == str else level
        if type(level) == int:
            dict_data1 = ind_name1[level_dict[level]].set_index(level_dict[level][0])[level_dict[level][1]].to_dict()
            dict_data2 = ind_name2[level_dict[level]].set_index(level_dict[level][0])[level_dict[level][1]].to_dict()
            dict_data = dict(dict_data2, **dict_data1)

        elif type(level) == list:
            dict_data = dict()
            for l in level:
                dict_data1 = ind_name1[level_dict[l]].set_index(level_dict[l][0])[level_dict[l][1]].to_dict()
                dict_data2 = ind_name2[level_dict[l]].set_index(level_dict[l][0])[level_dict[l][1]].to_dict()
                dict_data = dict(dict_data, **dict_data1)
                dict_data = dict(dict_data, **dict_data2)
        else:
            raise ValueError("Use SW_new must use level as int or str ")
        return dict_data
    else:
        ind_name = pd.read_excel(base_address + '行业分类.xlsx', sheet_name=ind_type)
        level = int(level) if type(level) == str else level
        if type(level) == int:
            dict_data = ind_name[level_dict[level]].set_index(level_dict[level][0])[level_dict[level][1]].to_dict()
        elif type(level) == list:
            df = pd.Series()
            for l in level:
                df = pd.concat(
                    [df, ind_name[level_dict[l]].set_index(level_dict[l][0])[level_dict[l][1]].drop_duplicates()])
            dict_data = df.to_dict()
        else:
            raise ValueError("level type should be int or list ")
        return dict_data
# （2）获取实时该行业是否正在使用
def get_useful_ind(ind_type,date_list):
    ind_close = get_daily_1factor('close', date_list=date_list, type=ind_type[:-1])
    if ind_type[:-1] == 'SW':
        before_ind = get_ind_con(ind_type[:-1],ind_type[-1])
        after_ind = get_ind_con(ind_type[:-1]+'2021', ind_type[-1])
        del_ind = ['801230.SI', '852311', '801231']

        before_ind = set(before_ind.keys()).difference(del_ind)
        after_ind = set(after_ind.keys()).difference(del_ind)

        ind_useful = pd.concat([ind_close[before_ind].loc[:20211210], ind_close[after_ind].loc[20211213:]])
    elif ind_type[:-1] == 'CITICS':
        ind = get_ind_con(ind_type[:-1], ind_type[-1])
        ind_useful = ind_close[ind]

    ind_useful = ~np.isnan(ind_useful.dropna(how='all',axis=1))

    return ind_useful


# Barra因子1：Size市值因子（和时间序列无关，截面因子）
def Barra_size(start_date,end_date,type='stock'):
    date_list = get_date_range(start_date,end_date)
    mv = get_daily_1factor('mkt_free_cap',date_list=date_list)
    if type == 'stock':
        return np.log(mv).loc[start_date:end_date]
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        return get_modified_ind_mv(date_list=None, code_list=None, ind_type=type).loc[start_date:end_date]
    else:
        ValueError('input type as stock ,SW, or CITICS')

# Barra因子2：Non-linear Size非线性市值（和时间序列无关，截面因子）
def Barra_Nonlinear_size(start_date,end_date,type='stock'):
    date_list = get_date_range(start_date,end_date)
    mv = get_daily_1factor('mkt_free_cap', date_list=date_list)
    if type == 'stock':
        stock_pool = clean_stock_list()
        log_mv = np.log(mv)[stock_pool]
        log_mv3 = log_mv ** 3

    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        log_mv = get_modified_ind_mv(date_list=None, code_list=None, ind_type=type)
        log_mv3 = log_mv ** 3
    else:
        ValueError('input type as stock ,SW, or CITICS')

    nonlinear_size = pd.DataFrame(index=log_mv3.index, columns=log_mv3.columns)
    for i in nonlinear_size.index:
        nonlinear_size.loc[i] = get_regression(log_mv.loc[i].dropna(), log_mv3.loc[i].dropna(), type='residual')

    return deal_data(nonlinear_size.astype(float)).loc[start_date:end_date]

# Barra因子3：Liquidity流动性，不过流动性因子因为需要过去252个交易日的信息，所以暂无（和时间序列有关，因为需要历史交易日）
def Barra_Liquidity(start_date,end_date,type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=12 * 21+1),end_date)
    if type == 'stock':
        free_turn = get_daily_1factor('free_turn', date_list=date_list)
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        float_A_shares = get_daily_1factor('float_A_shares', date_list=date_list)
        vol = get_daily_1factor('volume', date_list=date_list)

        free_turn = trans_factor_to_ind(vol,ind=type) / trans_factor_to_ind(float_A_shares,ind=type)
    else:
        ValueError('input type as stock ,SW, or CITICS')

    STOM = np.log(free_turn.rolling(21).sum())
    STOQ = np.log(free_turn.rolling(3 * 21,min_periods=21).sum())
    STOA = np.log(free_turn.rolling(12 * 21,min_periods=21).sum())

    return (0.35 * STOM + 0.35 * STOQ + 0.3 * STOA).loc[start_date:end_date]

# Barra因子4：Beta贝塔因子（和时间序列有关，因为需要历史252个交易日；但历史均能回补，不影响）
def Barra_Beta(start_date, end_date, type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date,offset=255),end_date)
    weight = get_half_weight(252,half_time=63)
    bench_pct = get_daily_1factor('pct_chg', date_list=date_list, type='bench')[['wind_A']].dropna()

    if type == 'stock':
        stock_pool = clean_stock_list(start_date=get_pre_trade_date(start_date,offset=255),end_date=end_date)
        pct_chg = get_daily_1factor('pct_chg', date_list=date_list)
        beta_result = ts_rolling_regression(bench_pct.iloc[1:], pct_chg.iloc[1:], rolling_days=252, weight=weight)[2]
        return beta_result.loc[start_date:end_date][stock_pool]
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_useful = get_useful_ind(type, date_list)
        ind_name = list(get_real_ind(ind_type=type[:-1], level=int(type[-1])).keys())
        pct_chg = get_daily_1factor('close', date_list=date_list,type=type[:-1])[ind_name].dropna(how='all').pct_change() * 100
        beta_result = ts_rolling_regression(bench_pct.iloc[1:], pct_chg.iloc[1:], rolling_days=252, weight=weight)[2]

        return beta_result.loc[start_date:end_date][ind_useful]
    else:
        ValueError('input type as stock ,SW, or CITICS')

# Barra因子5：Momentum动量因子，长期动量-短期动量（和时间序列有关，因为需要历史252个交易日；但历史均能回补，不影响）
def Barra_Momentum(start_date, end_date, type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date,offset=504+21),end_date)
    weight = get_half_weight(504-21, half_time=126)
    if type == 'stock':
        pct_chg = get_daily_1factor('pct_chg', date_list=date_list)
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_name = list(get_real_ind(ind_type=type[:-1],level=int(type[-1])).keys())
        pct_chg = get_daily_1factor('close', date_list=date_list, code_list=ind_name, type=type[:-1]).pct_change() * 100
    else:
        ValueError('input type as stock ,SW, or CITICS')

    ln_pct = np.log(1 + pct_chg / 100)

    momentum_data = pd.DataFrame(index=ln_pct.index, columns=ln_pct.columns)
    for i in range(504,len(ln_pct)):
        momentum_data.iloc[i] = ln_pct.iloc[i-504:i-21].mul(weight,axis=0).sum()

    momentum_data = momentum_data[~np.isnan(ln_pct)].dropna(how='all').dropna(how='all',axis=1)

    return momentum_data.loc[start_date:end_date]

# Barra因子6：Residual Volatility残差波动率（和时间序列有关，因为需要历史252个交易日；但历史均能回补，不影响）
def DASTD(start_date, end_date,type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=255), end_date)
    weight = get_half_weight(252, half_time=63)
    bench_pct = get_daily_1factor('pct_chg', date_list=date_list, type='bench')[['wind_A']]

    if type == 'stock':
        pct_chg = get_daily_1factor('pct_chg', date_list=date_list)
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_name = list(get_real_ind(ind_type=type[:-1], level=int(type[-1])).keys())
        pct_chg = get_daily_1factor('close', date_list=date_list,type=type[:-1])[ind_name].pct_change() * 100
    else:
        ValueError('input type as stock ,SW, or CITICS')
    excess_pct = pct_chg.sub(bench_pct['wind_A'],axis=0).dropna(how='all',axis=1)

    dastd = pd.DataFrame(index=excess_pct.index, columns=excess_pct.columns)
    for i in range(252,len(excess_pct)):
        dastd.iloc[i] = np.sqrt(((excess_pct.iloc[i-252:i] - excess_pct.iloc[i-252:i].mean()) ** 2).mul(weight,axis=0).sum())

    dastd = dastd[~np.isnan(excess_pct)].dropna(how='all').dropna(how='all', axis=1)
    return dastd.loc[start_date:end_date]

def CMRA(start_date, end_date,type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=255), end_date)
    bench_close = get_daily_1factor('close', date_list=date_list, type='bench')[['wind_A']]
    if type == 'stock':
        close = get_daily_1factor('close_badj', date_list=date_list)
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_name = list(get_real_ind(ind_type=type[:-1], level=int(type[-1])).keys())
        close = get_daily_1factor('close', date_list=date_list, type=type[:-1])[ind_name]
    else:
        ValueError('input type as stock ,SW, or CITICS')

    excess_pct = close.pct_change(21,fill_method=None).sub(bench_close['wind_A'].pct_change(21,fill_method=None),axis=0)
    excess_pct = np.log(1+excess_pct).dropna(how='all')

    crma = pd.DataFrame(index=excess_pct.index, columns=excess_pct.columns)
    for i in range(252,len(excess_pct)):
        crma.iloc[i] = np.log(1+excess_pct.iloc[i - 21 * 12:i][::-1][::21].cumsum().max()) - np.log(1 + excess_pct.iloc[i - 21 * 12:i][::-1][::21].cumsum().min())
    return crma.loc[start_date:end_date].dropna(how='all',axis=1)

def HSIGMA(start_date, end_date,type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=255), end_date)
    weight = get_half_weight(252, half_time=63)
    bench_pct = get_daily_1factor('pct_chg', date_list=date_list, type='bench')[['wind_A']].dropna()
    if type == 'stock':
        pct_chg = get_daily_1factor('pct_chg', date_list=date_list).dropna(how='all')
        hsigma_result = ts_rolling_regression(bench_pct, pct_chg, rolling_days=252, weight=weight)[3]
        return hsigma_result.loc[start_date:end_date].dropna(how='all', axis=1)

    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_name = list(get_real_ind(ind_type=type[:-1], level=int(type[-1])).keys())
        pct_chg = get_daily_1factor('close', date_list=date_list, code_list=ind_name, type=type[:-1]).pct_change(fill_method=None) * 100
        pct_chg = pct_chg.dropna(how='all')

        hsigma_result = ts_rolling_regression(bench_pct.iloc[1:], pct_chg, rolling_days=252, weight=weight)[3]

        return hsigma_result.loc[start_date:end_date].dropna(how='all', axis=1)
    else:
        ValueError('input type as stock ,SW, or CITICS')

def Barra_ResidualVolatility(start_date, end_date,type='stock'):
    dastd = DASTD(start_date, end_date, type=type)
    cmra = CMRA(start_date, end_date, type=type)
    hsigma = HSIGMA(start_date, end_date, type=type)
    residual_volatility = 0.74 * deal_data(dastd.astype(float)) + 0.16 * deal_data(cmra.astype(float)) \
                          + 0.10 * deal_data(hsigma.astype(float))
    # 数据解雇哦出来后，还需要对beta因子和size因子回归去除共线性
    size = Barra_size(start_date, end_date, type=type)
    beta = Barra_Beta(start_date, end_date, type=type)

    if (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_useful = get_useful_ind(type,get_date_range(start_date,end_date))
        residual_volatility = residual_volatility[ind_useful].dropna(how='all')
        size = size[ind_useful].loc[residual_volatility.index]
        beta = beta[ind_useful].loc[residual_volatility.index]

    new_residual_volatility = pd.DataFrame(index=residual_volatility.index, columns=residual_volatility.columns)
    for i in residual_volatility.index:
        y = residual_volatility.loc[i].dropna().astype(float)
        x = pd.concat([size.loc[i].rename('size'),beta.loc[i].rename('beta')],axis=1).dropna().astype(float)

        code_list = set(y.index).intersection(set(x.index))
        x,y = x.loc[code_list], y.loc[code_list]

        new_residual_volatility.loc[i] = get_regression(x, y, type='residual')

    return new_residual_volatility

# Barra因子7：BTOP账面市值比因子（和时间序列无关，截面因子）
def Barra_BTOP(start_date, end_date,type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    mv = get_daily_1factor('mkt_cap_ard',date_list=date_list).loc[start_date:end_date] # 公司市值
    # 获取公司每个季度的净资产
    total_asset = get_single_quarter('TOT_ASSETS', 'AShareBalanceSheet', report_type = '408001000',date_list=date_list)
    total_liabality = get_single_quarter('TOT_LIAB', 'AShareBalanceSheet', report_type='408001000', date_list=date_list)

    total_asset = fill_quarter2daily_by_fixed_date(total_asset, keep='last')
    total_liabality = fill_quarter2daily_by_fixed_date(total_liabality, keep='last')
    net_asset = (total_asset - total_liabality).loc[start_date:end_date]

    if type == 'stock':
        BTOP = net_asset / (mv * 1000)

    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_net_asset = trans_factor_to_ind(net_asset,ind=type)
        ind_mv = trans_factor_to_ind(mv,ind=type)
        BTOP = ind_net_asset / (ind_mv * 1000)
    else:
        ValueError('input type as stock ,SW, or CITICS')
    return BTOP

# Barra因子8：Leverage杠杆因子（和时间序列无关，截面因子）
def Barra_Leverage(start_date, end_date):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=120), end_date)
    mlev = get_single_quarter('S_FA_ASSETSTOEQUITY', 'AShareFinancialIndicator', report_type = '408001000',date_list=date_list)
    debt_to_asset = get_single_quarter('S_FA_DEBTTOASSETS', 'AShareFinancialIndicator', report_type='408001000',date_list=date_list)

    return deal_data(mlev) + deal_data(debt_to_asset)

# Barra因子9：Earning yeild盈利预期因子（价值因子）
def PE_TTM(start_date, end_date,type='stock'):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    earning = get_ttm_quarter('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type='408002000',date_list=date_list)
    earning = fill_quarter2daily_by_fixed_date(earning).loc[start_date:end_date]
    mv = get_daily_1factor('mkt_cap_ard', date_list=date_list).loc[start_date:end_date]  # 公司市值
    if type == 'stock':
        PE = (mv * 1000) / earning
    elif (type[:2] == 'SW') or (type[:6] == 'CITICS'):
        ind_mv = trans_factor_to_ind(mv, ind=type)
        ind_earning = trans_factor_to_ind(earning,ind=type)
        PE = (ind_mv * 1000) / ind_earning
    else:
        ValueError('input type as stock ,SW, or CITICS')

    return PE

def Barra_Earning_yeild(start_date, end_date):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=300), end_date)
    ep = 1/PE_TTM(start_date, end_date)
    # 未来12个月的预测EP
    table = 'AShareConsensusRollingData'
    factor_str = 'EST_PE,S_INFO_WINDCODE,EST_DT,ROLLING_TYPE'
    sql = r"select %s from wind.%s a where a.EST_DT >= '%s' and a.ROLLING_TYPE = 'FTTM'" % (factor_str, table, str(start_date))
    EST_PE = pd.read_sql(sql, con)
    est_pe = EST_PE.pivot_table(index='EST_DT',columns='S_INFO_WINDCODE',values='EST_PE')
    est_pe.index = est_pe.index.astype(int)
    est_pe.columns = pd.Series(est_pe.columns).apply(lambda x:trans_windcode2int(x))
    est_ep = 1/est_pe.reindex(date_list).loc[start_date:end_date]
    est_ep = est_ep[set(est_ep.columns).intersection(set(ep.columns))]
    # 经营现金流/市值
    net_cash_flow = get_ttm_quarter('NET_CASH_FLOWS_OPER_ACT', 'AShareCashFlow', report_type='408002000', date_list=date_list)
    net_cash_flow = fill_quarter2daily_by_fixed_date(net_cash_flow).loc[start_date:end_date]
    mv = get_daily_1factor('mkt_cap_ard', date_list=date_list).loc[start_date:end_date]  # 公司市值
    cetop = net_cash_flow / (mv * 1000)

    earning_yeild = deal_data(ep) + deal_data(est_ep).fillna(0) + deal_data(cetop)

    return earning_yeild


# Barra因子10：Growth（成长因子）
def _trans_financial2fixed_date(date):
    # 金融数据日期对齐
    date = trans_datetime2int(date)
    year, md = divmod(date, 10000)
    if md == 331:
        _md = 430
    elif md == 630:
        _md = 831
    elif md == 930:
        _md = 1031
    elif md == 1231:
        _md = 430
        year += 1
    else:
        raise ValueError("date must be financial report date")
    return int(year * 10000 + _md)

def get_history_real_value(df):
    _df = df.copy().sort_index()
    _df.index = _df.index.map(_trans_financial2fixed_date)
    _df = _df[~_df.index.duplicated(keep='last')].replace(np.nan, np.inf)
    _df = _df.reindex(get_date_range(_df.index[0],_df.index[-1])).bfill().replace(np.inf, np.nan)

    return _df

def Barra_Growth(start_date, end_date):
    date_list = get_date_range(get_pre_trade_date(start_date, offset=500), end_date)
    code_list = get_code_list()
    # 1、SGRO：年度同比营业收入增长
    SGRO = get_quarter_1factor('OPER_REV', 'AShareIncome', report_type = '408001000', date_list=date_list)
    SGRO = SGRO[((SGRO.index % 10000) // 100).isin([3,6,9,12])]
    sgro = get_yoy(SGRO)
    sgro = fill_quarter2daily_by_fixed_date(sgro).loc[start_date:end_date]
    # 2、EGRO：年度同比盈利增长
    EGRO = get_quarter_1factor('NET_PROFIT_EXCL_MIN_INT_INC', 'AShareIncome', report_type = '408001000', date_list=date_list)
    EGRO = EGRO[((EGRO.index % 10000) // 100).isin([3, 6, 9, 12])]
    egro = get_yoy(EGRO)
    egro = fill_quarter2daily_by_fixed_date(egro).loc[start_date:end_date]
    # 3、SGRO——con：分析师一致预期营业收入增长
    # 4、EGRO——con：分析师一致预期盈利增长
    table = 'AShareConsensusRollingData'
    factor_str = 'EST_OPER_REVENUE, NET_PROFIT, S_INFO_WINDCODE, EST_DT, ROLLING_TYPE'
    sql = r"select %s from wind.%s a where a.EST_DT >= '%s' and a.ROLLING_TYPE = 'FTTM'" % (factor_str, table, str(start_date))
    con_data = pd.read_sql(sql, con)

    SGRO_con = con_data.pivot_table(index='EST_DT',columns='S_INFO_WINDCODE',values='EST_OPER_REVENUE')
    EGRO_con = con_data.pivot_table(index='EST_DT',columns='S_INFO_WINDCODE',values='NET_PROFIT')
    SGRO_con.index,SGRO_con.columns  = SGRO_con.index.astype(int), pd.Series(SGRO_con.columns).apply(lambda x:trans_windcode2int(x))
    EGRO_con.index, EGRO_con.columns = EGRO_con.index.astype(int), pd.Series(EGRO_con.columns).apply(lambda x: trans_windcode2int(x))
    SGRO_con = SGRO_con[set(code_list).intersection(set(SGRO_con.columns))].reindex(date_list).loc[start_date:end_date]
    EGRO_con = EGRO_con[set(code_list).intersection(EGRO_con.columns)].reindex(date_list).loc[start_date:end_date]

    compare_SGRO = get_history_real_value(SGRO).loc[start_date:end_date]
    compare_EGRO = get_history_real_value(EGRO).loc[start_date:end_date]

    sgro_con = SGRO_con / compare_SGRO
    egro_con = EGRO_con / compare_EGRO

    growth = deal_data(sgro) + deal_data(egro) + deal_data(sgro_con).fillna(0) + deal_data(egro_con).fillna(0)

    return growth


