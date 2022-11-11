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

    factor = factor[ind_useful].loc[start_date:end_date].astype(float)

    return factor

start_date, end_date,ind = 20150101,20210630,'SW1'
factor = factor_test(Factor_new,start_date,end_date,ind)

factor_name = 'Factor_new'
test_result, value_result, ic, rank_ic = factor_in_box(factor,factor_name, ind,fee=0.001)
test_result
value_result[[1,2,3,4,5]]


# 检查一下所有的因子，是否具有np.inf和-np.inf


factor_name = 'Factor_residual_overnight'
start_date, end_date,ind = 20150101,20221028,'SW1'
factor = factor_test(Factor_residual_overnight,start_date,end_date,ind)

test_result, value_result, ic, rank_ic = factor_in_box(factor,factor_name, ind,fee=0.001)
test_result
value_result[[1,2,3,4,5]]
