import time,os,matplotlib,warnings
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *

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
    ind_codes = list(get_ind_con(ind_type[:-1],level=int(ind_type[-1])).keys())

    ind_result = np.r_['0,3', tuple(ind == x for x in ind_codes)]
    ind_mv = np.einsum('ijk,jk -> ijk',ind_result,mv)

    modified_ind_mv = pd.DataFrame(np.nansum(ind_mv,axis=2),index=ind_codes,columns=date_list).T

    return np.log(modified_ind_mv).replace([np.inf,-np.inf], np.nan)
#(4)获取回归系数，和回归残差
def get_regression(x,y):

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

    new_df = pd.concat([pd.Series(get_regression(mv.loc[date].dropna().values,df.loc[date].dropna().values)[0],index=df.loc[date].dropna().index).rename(date)
               for date in date_list],axis=1).T

    return new_df

    # 对市值和因子值进行中性化

#(1)获取板块内个股数量
def get_ind(ind='SW1',date_list=None):
    ind_code = get_daily_1factor(ind, date_list)
    ind_num = ind_code.T.apply(lambda x:x.value_counts()).T
    return ind_code,ind_num
# 3、将个股因子转换为行业整体因子
def transfactor_code2ind(df,ind='SW1', weight='same' ,way = 'mean'):
    df = df.dropna(how='all')
    date_list = df.index.to_list()
    code_list = df.columns.to_list()
    ind_code,ind_num = get_ind(ind=ind, date_list=date_list)
    inde_name = list(get_ind_con(ind[:-1],int(ind[-1])).keys())



    get_daily_1factor('pre_close',date_list=date_list,code_list=inde_name,type='SW')
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
def factor_in_box(factor,factor_name, ind='SW1',fee=0.04, save_path = 'E:/FactorTest/useful_factor/',factor_path = 'E:/FactorTest/'):
    factor = factor.dropna(how='all')

    test_start_date = max(20140101, factor.index[0])
    test_end_date = min(20211130, factor.index[-1])
    self = FactorTest(test_start_date=test_start_date, test_end_date=test_end_date, ind=ind, day=20,fee=fee)
    box_in, test_result, value_result = self.cal_factor_result(factor, save_path =None)

    if box_in == True:
        print('因子通过测试',box_in)
        factor_list = [x[:-4] for x in os.listdir(save_path)]
        flag = 1
        for old_factor in factor_list:
            other_factor = pd.read_pickle(save_path + old_factor + '.pkl')
            corr = factor.corrwith(other_factor,axis=1)
            if abs(corr.mean()) > 0.6:
                print(corr.mean())
                # 如果相关系数过高，则看多头收益率和多空收益率谁大
                box_in1, test_result1, value_result1 = self.cal_factor_result(other_factor, save_path=None)
                # 如果ic/rank_ic,icir/rank_icir,top_return,excess_return,ls_return五个有三个以上优秀，就选新的
                Better_IC = (test_result.loc[['ic','rank_ic'],'all'] > test_result1.loc[['ic','rank_ic'],'all']).max() == True
                Better_ICIR = (test_result.loc[['ICIR','rank_ICIR'],'all'] >test_result1.loc[['ICIR','rank_ICIR'],'all']).max() == True
                Better_Return = (test_result.loc[['top_return','excess_return','ls_return'],'all'] >
                              test_result1.loc[['top_return','excess_return','ls_return'],'all']).sum()

                if (Better_IC + Better_ICIR +Better_Return) <3:
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

    return test_result, value_result


class FactorTest(object):
    def __init__(self,day=20,ind='SW1',test_start_date = 20150101,test_end_date = 20201231,fee=0.04):
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
        code_list = get_ind_con(ind[:-1],int(ind[-1]))
        self.ind_list = code_list
        ind_open = get_daily_1factor('open', date_list=get_date_range(test_start_date, get_pre_trade_date(test_end_date,offset=-30)), code_list=code_list,type=ind[:-1])

        bench_open = get_daily_1factor('open',date_list=get_date_range(test_start_date, get_pre_trade_date(test_end_date,offset=-30)),
                                       code_list=['HS300','ZZ500','wind_A'],type='bench')

        self.ind_open = ind_open
        self.bench_open = bench_open

        ind_trade_profit = ind_open.loc[trade_date_list].pct_change()
        ind_trade_profit.index = pd.Series(ind_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))

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

        test_factor = self.deal_factor(new_factor)
        #test_factor = new_factor.copy()
        # 计算1：ic，rank_ic
        ic = test_factor.shift(1).corrwith(self.ind_trade_profit, axis=1)
        rank_ic = test_factor.shift(1).rank(pct=True, axis=1).corrwith(self.ind_trade_profit.rank(pct=True, axis=1), axis=1)
        # 计算2：净值曲线：top组，多空组
        ascending = False if rank_ic.mean() > 0 else True
        group_dict = self.get_group_factor(factor=test_factor, direction=ascending, group=group)  # 获取分组

        top_ind, bottom_ind = group_dict[1], group_dict[group]
        top_turn = (top_ind.astype(int).diff() == 1).sum(axis=1) / top_ind.sum(axis=1)  # 头部的换手率
        top_pct = self.ind_trade_profit[top_ind.shift(1)].loc[self.period_date_list]                   # 头部的周期收益率
        bottom_turn = (bottom_ind.astype(int).diff() == 1).sum(axis=1) / bottom_ind.sum(axis=1) # 尾部换手率
        bottom_pct = self.ind_trade_profit[bottom_ind.shift(1)].loc[self.period_date_list]             # 尾部的周期收益率

        # 获取多头，多空，benchmarK收益率
        top_pct_mean, bottom_pct_mean = (1 + top_pct.mean(axis=1))* (1 - top_turn * self.fee) -1, \
                                        (1 + bottom_pct.mean(axis=1))* (1 - bottom_turn * self.fee) -1 # 头部收益率，尾部收益率
        benchmark_pct = self.ind_trade_profit.loc[self.period_date_list].mean(axis=1) # 基准收益率
        top_net_value, bottom_net_value = (1 + top_pct_mean).cumprod(), (1 + bottom_pct_mean).cumprod() # 头部净值， 尾部净值
        benchmark_net_value =  (1 + benchmark_pct).cumprod() # 基准净值

        excess_pct_mean = top_pct_mean - benchmark_pct   # 超额收益
        ls_pct_mean = top_pct_mean - bottom_pct_mean     # 多空收益


        # 输出结果1：分阶段统计数据
        year_list = sorted(list(set([x // 10000 for x in self.period_date_list[1:]])))
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
            rank_day_direction = ((test_result.loc['rank_ic'].drop('all') > 0) == ic_direction).sum() / len(
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
                        if test_result.loc[['top_winrate', 'excess_winrate'], 'all'].min() > 0.5:
                            # 6、icir的绝对值必须＞1
                            if abs(test_result.loc[['ICIR','rank_ICIR'], 'all'].min()) > 0.5:
                                # 7、分组收益比率是第一组≥第2,3,4组≥5组
                                if ((value_result.loc['annual_return', 1] - value_result.loc['annual_return', [2, 3, 4]] > 0).sum() >=2) & \
                                    ((value_result.loc['annual_return', [2, 3, 4]] - value_result.loc['annual_return', 5] > 0).sum() >= 2):
                                    print(True)
                                    return True, test_result, value_result
        return False, test_result, value_result

    ############################################# 策略测试部分 ########################################################
    def single_factor_test(self,test_factor,fee):
        net_value = pd.Series(index=get_date_range(test_factor.index[0],get_pre_trade_date(test_factor.index[-1],offset=-1)))  # 计算净值
        turn = pd.Series(index=test_factor.index)
        base_money = 1 * (1 - fee)
        for i in range(0, len(test_factor) - 1):
            signal_date, next_signal_date = test_factor.index[i], test_factor.index[i + 1]
            signal = test_factor.loc[signal_date] * base_money
            cash = round(base_money - signal.sum(), 4)  # 购买之后剩余的现金

            buy_date, sell_date = get_pre_trade_date(signal_date, -1), get_pre_trade_date(next_signal_date, -1)
            pct_daily = self.ind_open.loc[buy_date:sell_date] / self.ind_open.loc[buy_date]

            month_net_value = (pct_daily * signal).sum(axis=1).iloc[1:]
            net_value.loc[month_net_value.index] = cash + month_net_value

            base_money = month_net_value.iloc[-1] + cash

            # 接下来考虑换手率对于现金的影响
            end_ind_weight = (pct_daily * signal).iloc[-1]
            next_ind_weight = test_factor.loc[next_signal_date] * base_money

            change_rate = (next_ind_weight - end_ind_weight)[(next_ind_weight - end_ind_weight) > 0].sum()
            turn.loc[next_signal_date] = change_rate

            base_money = base_money * (1 - change_rate * fee)

        net_value = net_value.iloc[1:]
        net_value.iloc[0] = 1

        return net_value, turn

    def strategy_test(self, factor, fee=0.004, save_path=None,save_name = 'strategy'):
        # 第一步：，信号生成：传入的dataframe有两种，当为True和Flase时，等权配置；当传入的dataframe为float时，则按照传入的权重进行配置，大于1时配置为1，小于1时不进行调整。
        if factor.values.dtype == bool:  # 表明等权配置
            factor = factor.reindex(self.period_date_list).dropna(how='all')  # 获取因子值
            test_factor = factor.div(factor.sum(axis=1), axis=0)
        elif factor.values.dtype in ('int64', 'int32', 'float64', 'float32'):
            factor = factor.reindex(self.period_date_list).dropna(how='all')  # 获取因子值
            test_factor = factor.copy()
            test_factor.loc[test_factor.sum(axis=1)[test_factor.sum(axis=1) > 1].index] = test_factor.loc[
                test_factor.sum(axis=1)[test_factor.sum(axis=1) > 0].index].div(
                test_factor.loc[test_factor.sum(axis=1)[test_factor.sum(axis=1) > 0].index].sum(axis=1), axis=0)
        else:
            ValueError("date must be bool, int or float")

        # 第二步：获取每个周期的持仓，固定时间点调仓：同时，需要考虑费率，即每次调仓时手续费计算为千4
        test_factor.fillna(0, inplace=True)
        net_value, turn = self.single_factor_test(test_factor,fee)

        # 第三步：获取每个基准的组合，benchmark：行业等权组合
        benchmark_factor = ~np.isnan(self.ind_open.loc[test_factor.index, test_factor.columns])
        benchmark_factor = benchmark_factor.astype(float).div(benchmark_factor.sum(axis=1), axis=0)
        benchmark_net_value, benchmark_turn = self.single_factor_test(benchmark_factor,fee)

        # 第四步：计算超额，超额的计算方式为，用两者净值相减
        bench_net_value = self.bench_open.loc[net_value.index] / self.bench_open.loc[net_value.index].iloc[0]
        all_net_value = pd.concat([net_value.rename('factor'),benchmark_net_value.rename('benchmark'),bench_net_value],axis=1)

        excess_value = pd.DataFrame(index=net_value.index,columns=['benchmark_excess', '300_excess', '500_excess', 'A_excess'])
        excess_value['benchmark_excess'] = net_value / benchmark_net_value
        excess_value[['300_excess','500_excess','A_excess']] = (net_value / bench_net_value[['HS300','ZZ500','wind_A']].T).T

        ########################################### 开始计算统计数据 #################################################################
        # 1、按固定周期计算单期结果：统计月度收益率，胜率，盈亏比，换手率。超额的胜率收益率，胜率，盈亏比，换手率
        period_profit = all_net_value.reindex(self.trade_date_list).dropna().pct_change()
        # （在固定的周期下）胜率，收益率，盈亏比，换手率
        year_list = sorted(list(set([x // 10000 for x in period_profit.index.to_list()[1:]])))
        result = pd.DataFrame(index=['胜率', '年化收益率', '盈亏比', '换手率',
                                     '相对基准胜率', '相对基准收益率', '相对基准盈亏比',
                                     '相对300胜率', '相对300收益率', '相对300盈亏比',
                                     '相对500胜率', '相对500收益率', '相对500盈亏比',
                                     '相对全A胜率', '相对全A收益率', '相对全A盈亏比'], columns=['all'] + year_list)

        for year in result.columns:
            year_date = net_value.index.to_list() if year == 'all' else net_value.loc[year * 10000 + 101:get_pre_trade_date((year+1) * 10000 + 101,offset=-1)].index.to_list()  # 日期列表
            #period_date = [get_pre_trade_date(x, offset=-1) for x in test_factor.loc[year_date[0]:year_date[-1]].index.to_list()]
            year_period_profit = period_profit.loc[year_date[0]:year_date[-1]]

            year_value = net_value.loc[year_date] / net_value.loc[year_date[0]]  # 给定周期的净值曲线
            year_benchmark_value = benchmark_net_value.loc[year_date] / benchmark_net_value.loc[year_date[0]] # 给定周期基准的净值曲线
            year_bench_value = bench_net_value.loc[year_date] / bench_net_value.loc[year_date[0]] # 给定周期指数的净值曲线

            result.loc['胜率',year] = (year_period_profit['factor']>0).sum() / len(year_period_profit['factor'])
            result.loc['年化收益率',year] = year_value.iloc[-1] ** (242/len(year_value)) - 1
            result.loc['盈亏比',year] = -year_period_profit['factor'][year_period_profit['factor']>0].mean() / year_period_profit['factor'][year_period_profit['factor']<0].mean()
            result.loc['换手率',year] = turn.loc[year_date[0]:year_date[-1]].mean()

            # 相对基准情况
            excess = (year_period_profit['factor'] - year_period_profit[['benchmark','HS300','ZZ500','wind_A']].T).T

            result.loc['相对基准胜率',year] = (excess['benchmark'] >0).sum() / len(excess)
            result.loc['相对基准收益率', year] = (excess_value['benchmark_excess'].loc[year_date[-1]] / excess_value['benchmark_excess'].loc[year_date[0]])** (242/len(year_value)) - 1
            result.loc['相对基准盈亏比', year] = -excess['benchmark'][excess['benchmark']>0].mean() / excess['benchmark'][excess['benchmark']<0].mean()

            result.loc['相对300胜率', year] = (excess['HS300'] > 0).sum() / len(excess)
            result.loc['相对300收益率', year] = (excess_value['300_excess'].loc[year_date[-1]] / excess_value['300_excess'].loc[year_date[0]]) ** (242 / len(year_value)) - 1
            result.loc['相对300盈亏比', year] = -excess['HS300'][excess['HS300'] > 0].mean() / excess['HS300'][excess['HS300'] < 0].mean()

            result.loc['相对500胜率', year] = (excess['ZZ500'] > 0).sum() / len(excess)
            result.loc['相对500收益率', year] = (excess_value['500_excess'].loc[year_date[-1]] / excess_value['500_excess'].loc[year_date[0]]) ** (242 / len(year_value)) - 1
            result.loc['相对500盈亏比', year] = -excess['ZZ500'][excess['ZZ500'] > 0].mean() / excess['ZZ500'][excess['ZZ500'] < 0].mean()

            result.loc['相对全A胜率', year] = (excess['wind_A'] > 0).sum() / len(excess)
            result.loc['相对全A收益率', year] = (excess_value['A_excess'].loc[year_date[-1]] / excess_value['A_excess'].loc[year_date[0]]) ** (242 / len(year_value)) - 1
            result.loc['相对全A盈亏比', year] = -excess['wind_A'][excess['wind_A'] > 0].mean() / excess['wind_A'][excess['wind_A'] < 0].mean()

            # 2、绘制净值曲线图：和单一策略的年化收益率，盈亏比，最大回撤，日胜率，

        result = result.astype(float).round(4)

        # 3、绘制超额收益曲线，相比基准，300，500，全A：年化超额收益率，盈亏比，最大回撤，日胜率
        if save_path != None:
            self.save_path = save_path
            self.draw_picture([all_net_value,excess_value],factor_name=save_name+'net_value')

            writer = pd.ExcelWriter(self.save_path + 'Strategy_' + save_name + str(self.day) + '.xlsx')
            result.to_excel(writer, sheet_name='result')
            all_net_value.to_excel(writer, sheet_name='net_value')
            excess_value.to_excel(writer, sheet_name='excess_value')
            writersheet = writer.sheets['net_value']
            writersheet.insert_image('G1', self.save_path + save_name + 'net_value'+str(self.day)+'.png')
            writer.close()


        return result, all_net_value,excess_value







