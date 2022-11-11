import time,os,matplotlib,warnings
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

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

#(3)获取行业市值
def get_modified_ind_mv(date_list=None, code_list=None, ind_type='SW1'):
    mv = get_daily_1factor('mkt_free_cap', date_list, code_list).dropna(how='all')
    date_list = mv.index.to_list()
    code_list = mv.columns.to_list()

    ind = get_daily_1factor(ind_type, date_list, code_list)
    ind_codes = list(get_real_ind(ind_type[:-1],level=int(ind_type[-1])).keys())

    ind_result = np.r_['0,3', tuple(ind == x for x in ind_codes)]
    ind_mv = np.einsum('ijk,jk -> ijk',ind_result,mv)

    modified_ind_mv = pd.DataFrame(np.nansum(ind_mv,axis=2),index=ind_codes,columns=date_list).T

    return np.log(modified_ind_mv).replace([np.inf,-np.inf], np.nan)
#(4)获取回归系数，和回归残差
def start_regression(x,y):

    lr = LinearRegression(fit_intercept=False)
    lr.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # 拟合
    y_predict = lr.predict(x.reshape(-1, 1))  # 预测

    return y- y_predict.reshape(1,-1)
# 2、进行市值中性化
def get_mv_neutral(df,type='SW1'):
    df = df.dropna(how='all')
    date_list = df.index.to_list()
    # type = stock表示，对个股因子进行市值中性化
    if type == 'stock':
        code_list = df.columns.to_list()
        mv = np.log(get_daily_1factor('mkt_free_cap', date_list=date_list, code_list=code_list, type='stock'))
    else: # type = 'SW1'
        mv = get_modified_ind_mv(date_list=date_list, code_list=None, ind_type=type)

    # 先把两个dataframe对其
    mv = mv[~np.isnan(df)]
    df = df[~np.isnan(mv)]

    new_df = pd.concat([pd.Series(start_regression(mv.loc[date].dropna().values,df.loc[date].dropna().values)[0],index=df.loc[date].dropna().index).rename(date)
               for date in date_list],axis=1).T

    return new_df

    # 对市值和因子值进行中性化

#(1)获取板块内个股数量
def get_ind(ind='SW1',date_list=None):
    ind_code = get_daily_1factor(ind, date_list)
    ind_num = ind_code.T.apply(lambda x:x.value_counts()).T
    return ind_code,ind_num
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

# 因子检查：即计算当日的因子，计算当月的因子
def factor_test(func,start_date,end_date,ind):
    end_date = int(datetime.datetime.now().strftime('%Y%m%d')) if end_date > int(datetime.datetime.now().strftime('%Y%m%d')) else end_date
    factor = func(start_date,end_date,ind).astype(float).round(5)
    factor1 = func(get_pre_trade_date(end_date,200),get_pre_trade_date(end_date,120),ind).astype(float).round(5)
    factor2 = func(get_pre_trade_date(end_date,120),end_date,ind).astype(float).round(5)

    date_list1,columns1 = factor1.index.to_list(),factor1.columns
    date_list2,columns2 = factor2.index.to_list(),factor2.columns

    ind_useful = get_useful_ind(ind,get_date_range(start_date,end_date))
    bad_factor1 = ((factor.loc[date_list1,columns1] == factor1.loc[date_list1,columns1])[ind_useful] == False).sum().sum()
    bad_factor2 = ((factor.loc[date_list2,columns2] == factor2.loc[date_list2,columns2])[ind_useful] == False).sum().sum()

    if ((factor == np.inf).sum().sum() >0) or ((factor == -np.inf).sum().sum() >0):
        print('数据存在np.inf,需要检查')


    if (bad_factor1 > len(date_list1)) | (bad_factor2 > len(date_list2)):
        print('数据存在问题，进行代码调整')
        print(bad_factor1, bad_factor2)
    else:
        print('因子无问题')
        print(bad_factor1,bad_factor2)

    return factor

# 3、将个股因子转换为行业整体因子
def transfactor_code2ind(df,ind='SW1', weight='same' ,way = 'mean'):
    df = df.dropna(how='all')
    date_list = df.index.to_list()
    code_list = df.columns.to_list()
    ind_code,ind_num = get_ind(ind=ind, date_list=date_list)
    inde_name = list(get_ind_con(ind[:-1],int(ind[-1])).keys())

    get_daily_1factor('pre_close',date_list=date_list,code_list=inde_name,type=ind[:-1])
    # weight为加权方式：即same为行业内个股等全，mv为市值加权
    if weight == 'same':
        weighted = pd.DataFrame(1, index = date_list, columns=code_list)
    elif weight == 'mv':
        weighted = np.log(get_daily_1factor('mkt_free_cap',date_list=date_list,code_list=code_list))

    ind_name = list(get_ind_con(ind[:-1], level=int(ind[-1])).keys())

    if way == 'mean':
        ind_factor = pd.concat([(df[ind_code == i] * weighted[ind_code == i].div(weighted[ind_code == i].sum(axis=1),axis=0)).sum(axis=1)
                               .rename(i) for i in ind_name],axis=1)
    elif way == 'std':
        ind_factor = pd.concat([(((df[ind_code == i] * weighted[ind_code == i].div(weighted[ind_code == i].sum(axis=1), axis=0)).mul((ind_code == i).sum(axis=1),axis=0))).std(axis=1)
                               .rename(i) for i in ind_name],axis=1)

    ind_factor = ind_factor[ind_num>0]

    return ind_factor

# 其他函数：
def trans_list_from_middle(need_list):
    # 把列表从中间往两侧重排列
    new_list = []
    while len(need_list)>0:
        new_list.append(need_list[len(need_list) // 2])
        need_list.remove(need_list[len(need_list) // 2])

    return new_list
# 入库因子的判断和比较
def factor_in_box(factor,factor_name, ind='SW1',fee=0.001, save_path = 'E:/FactorTest/useful_factor/',factor_path = 'E:/FactorTest/'):
    factor = factor.dropna(how='all')

    test_start_date = max(20140101, factor.index[0])
    test_end_date = min(20221130, factor.index[-1])
    self = FactorTest(test_start_date=test_start_date, test_end_date=test_end_date, ind=ind, day=20,fee=fee)
    box_in, test_result0, value_result0, ic0, rank_ic0 = self.cal_factor_result(factor, save_path ='E:/FactorTest/')

    if box_in == True:
        print('因子通过测试',box_in)
        factor_list = [x[:-4] for x in os.listdir(save_path)]
        flag = 1
        for old_factor in factor_list:
            other_factor = pd.read_pickle(save_path + old_factor + '.pkl')
            date_list = sorted(list(set(other_factor.index).intersection(set(factor.index))))
            other_factor = other_factor.loc[date_list]
            test_factor = factor.loc[date_list]
            corr = test_factor.corrwith(other_factor,axis=1)
            if abs(corr.mean()) > 0.6:
                print(corr.mean())
                box_in, test_result, value_result, ic, rank_ic = self.cal_factor_result(test_factor, save_path=None)
                # 如果相关系数过高，则看多头收益率和多空收益率谁大
                box_in1, test_result1, value_result1,ic1, rank_ic1 = self.cal_factor_result(other_factor, save_path=None)
                # 如果ic/rank_ic,icir/rank_icir,top_return,excess_return,ls_return五个有三个以上优秀，就选新的
                Better_IC = abs(test_result.loc[['ic','rank_ic'],'all'].mean()) > abs(test_result1.loc[['ic','rank_ic'],'all'].mean())
                Better_ICIR = abs(test_result.loc[['ICIR','rank_ICIR'],'all'].mean()) > abs(test_result1.loc[['ICIR','rank_ICIR'],'all'].mean())
                Better_Return = (test_result.loc[['top_return','excess_return','ls_return'],'all'] >
                              test_result1.loc[['top_return','excess_return','ls_return'],'all']).sum()

                if ((Better_IC + Better_ICIR) == 0) | ((Better_IC + Better_ICIR +Better_Return) <3):
                    flag = 0
                    print('测试因子和%s因子相关性过高,且表现不佳，予以删除' % old_factor )
                    break
                else:
                    print('测试因子和%s因子相关性过高,但表现较高，旧因子予以删除' % old_factor)
                    other_factor.to_pickle(factor_path + '/old_useful_factor/'+ old_factor + '.pkl')
                    os.remove(factor_path + '/useful_factor/'+ old_factor + '.pkl')

        if flag == 1:
            factor.to_pickle(factor_path + '/useful_factor/'+ factor_name + '.pkl')
            print('有效因子已保存')
    else:
        print('因子未通过测试', box_in)

    return test_result0, value_result0, ic0, rank_ic0

class FactorTest(object):
    def __init__(self,day=20,ind='SW1',test_start_date = 20150101,test_end_date = 20201231,fee=0.001):
        self.ind = ind
        self.day = day
        self.fee = fee
        # 获取因子和因子值
        test_date_list = get_date_range(get_pre_trade_date(test_start_date), test_end_date)
        start_date, end_date = test_date_list[0], test_date_list[-1]
        if day == 5:
            period_date_list = get_date_range(start_date, end_date, period='W')
        elif day == 10:
            period_date_list = get_date_range(start_date, end_date, period='W')
            period_date_list = period_date_list[::2]
        elif day == 20:
            period_date_list = get_date_range(start_date, end_date, period='M')
        self.period_date_list = period_date_list

        trade_date_list = [get_pre_trade_date(x, offset=-1) for x in period_date_list] # 当天收盘有结果，第二日开盘交易，所以交易日是下一天

        self.test_date_list = test_date_list
        self.period_date_list = period_date_list
        self.trade_date_list = trade_date_list

        # 对标的收益率
        code_list = get_real_ind(ind[:-1],int(ind[-1]))
        self.ind_list = code_list
        ind_open = get_daily_1factor('open', date_list=get_date_range(test_start_date, get_pre_trade_date(test_end_date,offset=-30)), code_list=code_list,type=ind[:-1])
        ind_close = get_daily_1factor('close', date_list=get_date_range(test_start_date,get_pre_trade_date(test_end_date, offset=-30)),code_list=code_list, type=ind[:-1])

        bench_open = get_daily_1factor('open',date_list=get_date_range(test_start_date, get_pre_trade_date(test_end_date,offset=-30)),
                                       code_list=['HS300','ZZ500','wind_A'],type='bench')
        self.ind_useful = get_useful_ind(ind, test_date_list)
        self.ind_open = ind_open
        self.ind_close = ind_close
        self.bench_open = bench_open

        ind_trade_profit = ind_open.loc[trade_date_list].pct_change()
        ind_trade_profit.index = pd.Series(ind_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))
        ind_trade_profit = ind_trade_profit[self.ind_useful]


        bench_trade_profit = bench_open.loc[trade_date_list].pct_change()
        bench_trade_profit.index = pd.Series(bench_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))

        self.ind_trade_profit = ind_trade_profit
        self.bench_trade_profit = bench_trade_profit

        # 指数收益率
        index_open = get_daily_1factor('open', date_list=get_date_range(test_start_date, get_pre_trade_date(test_end_date,offset=-30)), code_list=['wind_A','HS300','ZZ500'], type='bench')
        index_trade_profit = index_open.loc[trade_date_list].pct_change()
        index_trade_profit.index = pd.Series(index_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))

        self.index_open = index_open
        self.index_trade_profit = index_trade_profit
    ############################################# 因子测试部分 ########################################################
    # 进行因子处理
    def deal_factor(self, factor):
        # 数据处理：标准化，中性化
        test_factor = deal_data(factor)
        test_factor = get_mv_neutral(test_factor, type=self.ind)

        return test_factor
    # 获取分组的组合
    def get_group_factor(self, factor, direction, group=5):
        choice_num = (~np.isnan(factor)).sum(axis=1) // group
        # 第一步：获取分组
        group_num = pd.DataFrame(index=choice_num.index, columns=range(1,group+1))
        group_num[1], group_num[group] = choice_num, choice_num # 首尾两组必须是一致的
        # 其余的部分往中间组填充
        average_num = ((~np.isnan(factor)).sum(axis=1) - 2 * choice_num) // (group - 2)
        mod_num = ((~np.isnan(factor)).sum(axis=1) - 2 * choice_num) %  (group - 2)
        for i in trans_list_from_middle(list(range(1,group+1))[1:-1]):
            group_num[i] = average_num + mod_num.apply(lambda x: min(1, x))
            mod_num = mod_num.apply(lambda x: max(x - 1, 0))
        group_num = group_num.cumsum(axis=1)

        factor_rank = factor.rank(axis=1, ascending=direction).T
        group_dict = dict()
        for i in range(1,group+1):
            if i == 1 :
                group_dict[i] = (factor_rank <= group_num[i]).T
            else:
                group_dict[i] = ((factor_rank > group_num[i-1]).T & (factor_rank <= group_num[i]).T)

        return group_dict
    # 画图
    def draw_picture(self,df_list,factor_name):
        fig = plt.subplots(figsize=(20, 15))

        for i in range(0,len(df_list)):
            df = df_list[i].copy()
            df.index = df.index.astype(str)

            exec('a'+str(i)+' = plt.subplot(len(df_list), 1, i+1)')
            exec('a'+str(i)+'.plot(df.index, df.values)')
            if i != len(df_list)-1:
                exec('a' + str(i) + '.set_xticks([])')
                exec('a' + str(i) + '.legend(df.columns)')

            else:
                xticks = list(range(0, len(df.index), 10))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
                xlabels = [df.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
                exec('a' + str(i) + '.set_xticks(xticks)')
                exec('a' + str(i) + '.legend(df.columns)')
                exec('a' + str(i) + '.set_xticklabels(xlabels, rotation=0, fontsize=10)')
                for tl in eval('a' + str(i) + '.get_xticklabels()'):
                    tl.set_rotation(90)

        plt.savefig(self.save_path + factor_name + str(self.day) + '.png')
    # 因子测试
    def factor_test(self,new_factor,group=5):
        #test_factor.columns = pd.Series(test_factor.columns).apply(lambda x:self.ind_list[x])
        period_date_list = sorted(list(set(new_factor.index).intersection(self.period_date_list)))
        test_factor = self.deal_factor(new_factor)
        # 计算1：ic，rank_ic
        ic = test_factor.shift(1).corrwith(self.ind_trade_profit, axis=1)
        rank_ic = test_factor.shift(1).rank(pct=True, axis=1).corrwith(self.ind_trade_profit.rank(pct=True, axis=1), axis=1)
        # 计算2：净值曲线：top组，多空组
        ascending = False if rank_ic.mean() > 0 else True
        group_dict = self.get_group_factor(factor=test_factor, direction=ascending, group=group)  # 获取分组

        top_ind, bottom_ind = group_dict[1], group_dict[group]
        top_turn = (top_ind.astype(int).diff() == 1).sum(axis=1) / top_ind.sum(axis=1)  # 头部的换手率
        top_pct = self.ind_trade_profit[top_ind.shift(1)].loc[period_date_list]                   # 头部的周期收益率
        bottom_turn = (bottom_ind.astype(int).diff() == 1).sum(axis=1) / bottom_ind.sum(axis=1) # 尾部换手率
        bottom_pct = self.ind_trade_profit[bottom_ind.shift(1)].loc[period_date_list]             # 尾部的周期收益率

        # 获取多头，多空，benchmarK收益率
        top_pct_mean, bottom_pct_mean = (1 + top_pct.mean(axis=1))* (1 - top_turn * self.fee) -1, \
                                        (1 + bottom_pct.mean(axis=1))* (1 - bottom_turn * self.fee) -1 # 头部收益率，尾部收益率
        benchmark_pct = self.ind_trade_profit.loc[period_date_list].mean(axis=1) # 基准收益率
        top_net_value, bottom_net_value = (1 + top_pct_mean).cumprod(), (1 + bottom_pct_mean).cumprod() # 头部净值， 尾部净值
        benchmark_net_value =  (1 + benchmark_pct).cumprod() # 基准净值

        excess_pct_mean = top_pct_mean - benchmark_pct   # 超额收益
        ls_pct_mean = top_pct_mean - bottom_pct_mean     # 多空收益

        # 输出结果1：分阶段统计数据
        year_list = sorted(list(set([x // 10000 for x in period_date_list[1:]])))
        test_result = pd.DataFrame(index=['ic', 'rank_ic', 'ICIR', 'rank_ICIR',
                                          'top_return', 'top_sharpe', 'top_turn', 'top_winrate', 'top_wlratio',
                                          'top_maxdown',
                                          'excess_return', 'excess_sharpe', 'excess_winrate', 'excess_wlratio',
                                          'excess_maxdown',
                                          'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown',
                                          ], columns=['all'] + year_list)
        for year in test_result.columns:
            year_date = test_factor.index.to_list() if year == 'all' else test_factor.loc[
                                                        year * 10000 + 101:year * 10000 + 1231].index.to_list()  # 日期列表

            test_result.loc['ic', year] = ic.loc[year_date].mean()
            test_result.loc['rank_ic', year] = rank_ic.loc[year_date].mean()
            test_result.loc['ICIR', year] = ic.loc[year_date].mean() / ic.loc[year_date].std() * np.sqrt(240 / self.day)
            test_result.loc['rank_ICIR', year] = rank_ic.loc[year_date].mean() / rank_ic.loc[year_date].std() * np.sqrt(240 / self.day)

            # 计算头部组合收益情况
            for name in ['top', 'excess', 'ls']:
                pct_mean = top_pct_mean.copy() if name == 'top' else excess_pct_mean.copy() if name == 'excess' else ls_pct_mean.copy()
                pct_mean = pct_mean.loc[set(pct_mean.index).intersection(year_date)].sort_index()
                net_value = (1 + pct_mean).cumprod()

                test_result.loc[name + '_return', year] = net_value.iloc[-1] ** ((240 / self.day) / len(year_date)) - 1
                test_result.loc[name + '_sharpe', year] = (pct_mean.mean() / pct_mean.std()) * np.sqrt(240 / self.day)
                if name == 'top':
                    test_result.loc[name + '_turn', year] = top_turn.loc[year_date].mean()

                test_result.loc[name + '_winrate', year] = (pct_mean > 0).sum() / len(pct_mean)
                test_result.loc[name + '_wlratio', year] = -pct_mean[pct_mean > 0].mean() / \
                                                           pct_mean[pct_mean < 0].mean()
                test_result.loc[name + '_maxdown', year] = ((net_value - net_value.cummax()) / net_value.cummax()).min()

        test_result = test_result.astype(float).round(4)

        # 输出结果2：净值曲线数据
        wind_A_net_value = (1 + self.index_trade_profit.loc[top_pct.index]['wind_A']).cumprod() # wind全A的净值
        exceess_net_value = (1 + excess_pct_mean).cumprod() # 超额j净值
        ls_net_value = (1 + ls_pct_mean).cumprod() # 多空净值

        # 5组收益率净值
        def get_pct_mean(x):
            top_turn = (x.astype(int).diff() == 1).sum(axis=1) / x.sum(axis=1)  # 头部的换手率
            top_pct = self.ind_trade_profit[x.shift(1)].loc[self.period_date_list]  # 头部的周期收益率
            pct_mean = (1 + top_pct.mean(axis=1))* (1 - top_turn * self.fee) -1

            return pct_mean.dropna(how='all')

        group_pct = pd.concat([get_pct_mean(group_dict[i]).rename(i) for i in group_dict.keys()],axis=1)
        group_net_value = (1 + group_pct).cumprod()

        # 合并起来做一个统计数据
        all_net_values = pd.concat([benchmark_net_value.rename('bench'), wind_A_net_value, group_net_value, exceess_net_value.rename('excess'), ls_net_value.rename('ls')], axis=1)
        all_net_values.iloc[0] = 1
        all_net_pct = all_net_values.pct_change()
        all_net_pct.iloc[0] = all_net_values.iloc[0] - 1

        value_result = pd.DataFrame(index=['annual_return', 'annual_sharpe', 'win_rate', 'ls_ratio', 'maxdown'],
                                    columns=all_net_values.columns)
        value_result.loc['annual_return'] = (all_net_values.iloc[-1] ** (242 /self.day / len(all_net_values))) - 1
        value_result.loc['annual_sharpe'] = all_net_pct.mean() / all_net_pct.std() * np.sqrt(242 / self.day)
        value_result.loc['win_rate'] = (all_net_pct > 0).sum() / len(all_net_pct)
        value_result.loc['ls_ratio'] = - all_net_pct[all_net_pct > 0].mean() / all_net_pct[all_net_pct < 0].mean()
        value_result.loc['maxdown'] = ((all_net_values - all_net_values.cummax()) / all_net_values.cummax()).min()

        value_result = value_result.astype(float).round(4)

        return test_result, value_result, all_net_values, ic, rank_ic
    # 开始进行因子测试
    def cal_factor_result(self,factor,save_path =None,factor_name = 'test_factor'):
        factor = factor[self.ind_useful]
        # 需要剔除的行业：综合
        need_del_list = ['801230.SI', '852311', '801231']
        factor = factor[factor.columns.difference(need_del_list)]

        factor = factor.reindex(self.period_date_list).dropna(how='all')  # 获取因子值
        test_result, value_result, all_net_values, ic, rank_ic = self.factor_test(factor)
        # 写入文件
        if save_path != None:
            self.save_path =save_path
            df0 = pd.concat([rank_ic.rolling(6).mean().rename('period_rank_ic'),ic.rolling(6).mean().rename('period_ic')],axis=1)
            df1 = all_net_values[[1, 5, 'bench', 'wind_A']].rename(columns={1: 'top', 5: 'bottom'})
            df2 = all_net_values[['excess', 'ls']]
            df3 = all_net_values[[1, 2, 3, 4, 5]]
            self.draw_picture([df0, df1, df2, df3], factor_name)
            writer = pd.ExcelWriter(self.save_path + 'Result_' + factor_name + str(self.day) + '.xlsx')
            test_result.to_excel(writer, sheet_name='factor')
            value_result.to_excel(writer, sheet_name='net_value')
            writersheet = writer.sheets['net_value']
            writersheet.insert_image('A7', self.save_path + factor_name + str(self.day) + '.png')
            writer.close()

        # 因子评估：从如下几个方面
        # 1、ic绝对值是不是＞0.01；如果低于0.01，则不通过
        if (abs(test_result.loc['ic', 'all']) >= 0.02) or (abs(test_result.loc['rank_ic', 'all']) >= 0.02):
            # 2、因子间隔两年的ic方向是否一致，如果超过超过一半的ic方向不一致，则剔除
            ic_direction = test_result.loc['ic', 'all'] > 0
            day_direction = ((test_result.loc['ic'].drop('all') > 0) == ic_direction).sum() / len(
                (test_result.loc['ic'].drop('all'))) > 0.5

            rank_ic_direction = test_result.loc['rank_ic', 'all'] > 0
            rank_day_direction = ((test_result.loc['rank_ic'].drop('all') > 0) == rank_ic_direction).sum() / len(
                (test_result.loc['ic'].drop('all'))) > 0.5

            if (day_direction | rank_day_direction):
                # 3、超额收益率 和 多空收益率必须＞0
                if (test_result.loc['excess_return', 'all'] > 0.01) & (test_result.loc['ls_return', 'all'] > 0)\
                        & (test_result.loc['ls_return', 'all'] > test_result.loc['excess_return', 'all']):
                    # 4、超额收益率和多空收益率＞0的年份必须超过一半
                    if (((test_result.loc[['excess_return', 'ls_return']].drop('all', axis=1) > 0).sum(axis=1) / (
                            len(test_result.columns) - 1)).min() >= 0.5) & \
                            (((test_result.loc[['excess_return', 'ls_return']].drop('all', axis=1) > 0).sum(axis=1) / (
                            len(test_result.columns) - 1)).max() > 0.5):
                        # 5、胜率必须超过50%
                        if test_result.loc[['ls_winrate', 'excess_winrate'], 'all'].min() > 0.5:
                            # 6、icir的绝对值必须＞1
                            if abs(test_result.loc[['ICIR','rank_ICIR'], 'all'].max()) > 0.5:
                                # 7、分组收益比率是第一组≥第2,3,4组≥5组
                                if ((value_result.loc['annual_return', 1] - value_result.loc['annual_return', [2, 3, 4]] > 0).sum() >=2) | \
                                    ((value_result.loc['annual_return', [2, 3, 4]] - value_result.loc['annual_return', 5] > 0).sum() >= 2):
                                    return True, test_result, value_result, ic, rank_ic

        return False, test_result, value_result, ic, rank_ic

    ############################################# 策略测试部分 ########################################################
    def single_factor_test(self,test_factor,fee):
        trade_date_list = get_date_range(test_factor.index[0], test_factor.index[-1],period='M')

        net_value = pd.Series(index=get_date_range(test_factor.index[0],get_pre_trade_date(test_factor.index[-1],offset=-1)))  # 计算净值
        turn = pd.Series(index=test_factor.index)
        base_money = 1 * (1 - fee)
        for i in range(0, len(trade_date_list) - 1):
            signal_date, next_signal_date = trade_date_list[i], trade_date_list[i + 1]
            buy_date, sell_date = get_pre_trade_date(signal_date, -1), get_pre_trade_date(next_signal_date, -1)
            month_factor = test_factor.loc[signal_date:next_signal_date].iloc[:-1].reindex(getData.get_date_range(signal_date,sell_date)).ffill()
            if month_factor.iloc[:-1].sum(axis=1).max() > 0:
                money_weight = base_money / len(month_factor.iloc[:-1].sum()[month_factor.iloc[:-1].sum()>0])  # 把钱分成几份
                month_factor = month_factor.replace(False,np.nan).ffill(limit=1).fillna(False)

                #pct_daily = (self.ind_open.loc[buy_date:sell_date] / self.ind_open.loc[buy_date])[month_factor].dropna(how='all',axis=1) * money_weight
                #pct_daily = (self.ind_open.loc[buy_date:sell_date][month_factor].pct_change(fill_method=None).dropna(how='all',axis=1).fillna(0) +1).cumprod() * money_weight

                pct_daily = (self.ind_open.loc[buy_date:sell_date][month_factor].pct_change(fill_method=None).dropna(
                    how='all', axis=1).fillna(0) + 1).cumprod() * money_weight

                daily_net_value = pct_daily.ffill().sum(axis=1)

                net_value.loc[pct_daily.index] = daily_net_value
            else:
                money_weight = 0
                pct_daily = (self.ind_open.loc[buy_date:sell_date] / self.ind_open.loc[buy_date])[month_factor] * money_weight
                daily_net_value = pct_daily.ffill().sum(axis=1)

                net_value.loc[pct_daily.index] = base_money

            base_money = net_value.loc[pct_daily.index].iloc[-1]  # 最后收盘时的总现金

            # 接下来考虑换手率对于现金的影响
            if test_factor.loc[next_signal_date].sum() > 0:
                next_ind_weight = test_factor.loc[next_signal_date] / test_factor.loc[next_signal_date].sum() * base_money
            else:
                next_ind_weight = pd.Series(0,end_ind_weight.index)
            end_ind_weight = pct_daily.iloc[-1].fillna(0)
            end_ind_weight = end_ind_weight.reindex(next_ind_weight.index).fillna(0)
            change_rate = (next_ind_weight - end_ind_weight)[(next_ind_weight - end_ind_weight) > 0].sum() / base_money
            turn.loc[next_signal_date] = change_rate

            base_money = base_money * (1 - change_rate * fee)

        net_value = net_value.iloc[1:]

        return net_value, turn

    # 画图
    def draw_strategy_picture(self, df,sav_path=None):
        df_list = df.copy()
        df_list.index = df_list.index.astype(str)
        fig = plt.subplots(figsize=(40, 15))
        ax1 = plt.subplot(2, 1, 1)

        ax1.plot(df_list.index, df_list[['factor','bench']].values)
        ax1.set_xticks([])
        ax1.legend(loc = 'best',labels = ['factor','bench'])
        ax2 =  plt.subplot(2, 1, 2)
        ax2.plot(df_list.index, df_list[['excess']].values)

        xticks = list(range(0, len(df_list.index), 20))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
        xlabels = [df_list.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
        ax2.set_xticks(xticks)
        ax2.legend(loc='best', labels=['excess'])
        ax2.set_xticklabels(xlabels, rotation=0, fontsize=20)
        for tl in ax2.get_xticklabels():
            tl.set_rotation(90)

        if sav_path != None:
            plt.savefig(sav_path + 'sentiment.jpg')
        plt.show()

    def strategy_test(self, factor, fee=0.001, save_path=None,save_name = 'strategy'):
        need_del_list = ['801230.SI', '852311', '801231']
        period_date_list = sorted(list(set(factor.index).intersection(self.period_date_list)))
        bench_mark = self.ind_useful[factor.columns.difference(need_del_list)].loc[factor.index]
        # 第一步：信号生成：传入的dataframe有两种，当为True和Flase时，等权配置；当传入的dataframe为float时，则按照传入的权重进行配置，大于1时配置为1，小于1时不进行调整。
        factor_net_value, factor_turn = self.single_factor_test(factor, fee)
        benchmark_net_value, benchmark_turn = self.single_factor_test(bench_mark, fee)

        excess_net_value = factor_net_value / benchmark_net_value # 超额净值
        top_pct = factor_net_value.loc[[get_pre_trade_date(x, -1) for x in period_date_list]].pct_change(1)  # 头部组合超额收益
        excess_pct = excess_net_value.loc[[get_pre_trade_date(x, -1) for x in period_date_list]].pct_change(1)  # 单期超额收益

        ########################################### 开始计算统计数据 ################################################################
        year_list = sorted(list(set([x // 10000 for x in period_date_list[1:]])))
        test_result = pd.DataFrame(index=['factor_return', 'factor_sharpe', 'factor_turn', 'factor_winrate', 'factor_wlratio','factor_maxdown',
                                          'excess_return', 'excess_sharpe', 'excess_winrate', 'excess_wlratio','excess_maxdown',], columns=['all'] + year_list)
        for year in test_result.columns:
            year_date = factor_net_value.index.to_list() if year == 'all' else factor_net_value.loc[get_pre_trade_date(year * 10000 + 101):get_pre_trade_date(year * 10000 + 1231)].index.to_list()  # 日期列表
            period_year_list = top_pct.index.to_list() if year == 'all' else \
                top_pct.loc[get_pre_trade_date(year * 10000 + 101, -2):get_pre_trade_date((year + 1) * 10000 + 101,-1)].index.to_list()

            # 计算组合收益率
            for name in ['factor', 'excess']:
                net_value = factor_net_value.loc[year_date].copy() if name == 'top' else excess_net_value.loc[year_date].copy()
                net_value = net_value / net_value.iloc[0]
                period_pct = top_pct.loc[period_year_list].copy() if name == 'top' else excess_pct.loc[period_year_list].copy()
                test_result.loc[name + '_return', year] = net_value.iloc[-1] ** (252 / len(year_date)) - 1
                test_result.loc[name + '_sharpe', year] = period_pct.mean() / period_pct.std() * np.sqrt(240 / self.day)
                if name == 'factor':
                    test_result.loc[name + '_turn', year] = factor_turn.mean() if year == 'all' else factor_turn.loc[get_pre_trade_date(year * 10000 + 101):get_pre_trade_date(year * 10000 + 1231)].mean()

                test_result.loc[name + '_winrate', year] = (period_pct > 0).sum() / len(period_pct)
                test_result.loc[name + '_wlratio', year] = -period_pct[period_pct > 0].mean() / period_pct[
                    period_pct < 0].mean()
                test_result.loc[name + '_maxdown', year] = ((net_value - net_value.cummax()) / net_value.cummax()).min()

        test_result = test_result.astype(float).round(4)

        all_net_value = pd.concat([factor_net_value.rename('factor'), benchmark_net_value.rename('bench'), excess_net_value.rename('excess')], axis=1)

        self.draw_strategy_picture(all_net_value, save_path)

        return test_result, all_net_value


    ############################################# 风险因子 #############################################################
    def risk_factor_test(self,factor,future_period = 'single'):
        risk_factor = factor.copy()
        if future_period == 'single':
            period_date_list = get_date_range(risk_factor.index[0],get_pre_trade_date(risk_factor.index[-1],-1,'M'),period='M')
            future_pct = self.ind_open.reindex([get_pre_trade_date(x,-1) for x in period_date_list]).dropna(how='all').pct_change(fill_method=None)
            future_pct.index = [get_pre_trade_date(x) for x in future_pct.index]
            future_pct = future_pct.shift(-1)
        else:
            future_pct = self.ind_open.pct_change(20,fill_method=None).shift(-21)

            risk_factor[risk_factor.rolling(10).sum() >1] = 0

        bench_pct = future_pct.mean(axis=1)

        result = pd.DataFrame(index = risk_factor.index,columns=['num','pct','excess','win_rate'])
        for date in (risk_factor.index):
            risk_ind = risk_factor.loc[date][risk_factor.loc[date]==True].index.to_list()
            if len(risk_ind) == 0:
                continue
            result.loc[date] = len(risk_ind), future_pct.loc[date,risk_ind].mean(), \
                                   future_pct.loc[date, risk_ind].mean()-bench_pct.loc[date], \
                                   (future_pct.loc[date,risk_ind] < bench_pct.loc[date]).sum() / len(risk_ind)
        result = result.astype(float).round(4)
        result = result.dropna(how='all')


        print('平均超额',round(result['excess'].mean(),4),'胜率',round(result['win_rate'].mean(),4))

        return result







