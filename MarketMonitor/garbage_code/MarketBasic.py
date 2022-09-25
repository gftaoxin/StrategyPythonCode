import pandas as pd
import numpy as np
import requests, json
from multiprocessing import Pool
import matplotlib as plt

import datetime, time, sys, os
from tqdm import tqdm
from dataApi import getData, tradeDate, stockList
import cx_Oracle
from BasicData.local_path import *
con = cx_Oracle.connect("windquery", "wind2010query", "10.2.89.132:1521/winddb", threaded=True) # 写入信号

# 该数据主要用来定义市场情绪的部分数据，包括板块数据，龙头股数据，涨停股，炸板股，强势股等等，用于第二日的数据 计算
class BasicData(object):
    ####先提取需要使用的基础数据####
    def __init__(self, start_date=20130101, end_date=20220901):
        date_list = getData.get_date_range(20100101, end_date)
        start_date = getData.get_date_range(start_date, end_date)[0]
        end_date = getData.get_date_range(start_date, end_date)[-1]
        date_list = date_list[date_list.index(start_date) - 250:]

        self.date_list = date_list
        self.start_date = start_date
        self.end_date = end_date
        ######日频数据：上市日期，收盘价，最高价，最低价，昨收价，复权收盘价，复盘最高价，换手率，指数收盘价##########
        ipo_date = getData.get_daily_1factor('live_days', date_list=date_list)
        close = getData.get_daily_1factor('close', date_list=date_list).dropna(how='all', axis=1)
        high = getData.get_daily_1factor('high', date_list=date_list).dropna(how='all', axis=1)
        low = getData.get_daily_1factor('low', date_list=date_list).dropna(how='all', axis=1)
        pre_close = getData.get_daily_1factor('pre_close', date_list=date_list).dropna(how='all', axis=1)
        close_adj = getData.get_daily_1factor('close_badj', date_list=date_list).dropna(how='all', axis=1)
        high_adj = getData.get_daily_1factor('high_badj', date_list=date_list).dropna(how='all', axis=1)
        close_adj.fillna(method='ffill', inplace=True)
        high_adj.fillna(method='ffill', inplace=True)
        turn = getData.get_daily_1factor('free_turn', date_list=date_list).dropna(how='all', axis=1)
        Index_close = getData.get_daily_1factor('close', type='bench')[['SZZZ', 'CYBZ']].shift(1).loc[start_date:end_date]
        Stock_Pct = getData.get_daily_1factor('pct_chg', date_list=date_list)
        amt = getData.get_daily_1factor('amt', date_list=date_list).dropna(how='all', axis=1)

        self.amt = amt
        self.ipo_date = ipo_date
        self.close = close
        self.high = high
        self.low = low
        self.pre_close = pre_close
        self.close_adj = close_adj
        self.high_adj = high_adj
        self.turn = turn
        self.Index_close = Index_close
        self.Stock_Pct = Stock_Pct

        ##########################################概念板块日间数据：###############################################
        code_str = 'F_INFO_WINDCODE,S_CON_WINDCODE,S_CON_INDATE,S_CON_OUTDATE'
        sql = r"select %s from wind.AINDEXMEMBERSWINDZL" % (code_str)
        stock_concept = pd.read_sql(sql, con)

        #######概念板块数据集合b0:wind行业指数  02:概念板块指数  62:万得全球行业分类标准
        code_str = 'S_INFO_INDEXCODE,S_INFO_NAME,S_INFO_INDUSTRYCODE,S_INFO_INDUSTRYCODE2'
        sql = r"select %s from wind.INDEXCONTRASTSECTOR" % (code_str)
        index_label = pd.read_sql(sql, con)

        index_label.set_index('S_INFO_INDEXCODE', inplace=True)
        concept_list = set(stock_concept['F_INFO_WINDCODE']).intersection(set(index_label.index))
        index_label = index_label.loc[concept_list]

        #######获取股票名称############
        code_str = 'S_INFO_WINDCODE,S_INFO_NAME'
        sql = r"select %s from wind.AShareDescription" % (code_str)
        stock_name = pd.read_sql(sql, con).set_index('S_INFO_WINDCODE')
        code_list = set(stock_concept['S_CON_WINDCODE']).intersection(set(stock_name.index))
        stock_name = stock_name.loc[code_list]

        stock_concept = stock_concept[(stock_concept['F_INFO_WINDCODE'].isin(concept_list)) & (stock_concept['S_CON_WINDCODE'].isin(code_list))]

        ######获取名字#####
        stock_concept['Name'] = stock_concept['F_INFO_WINDCODE'].apply(lambda x: index_label.loc[x, 'S_INFO_NAME'])
        stock_concept['Ind_Code'] = stock_concept['F_INFO_WINDCODE'].apply(lambda x: index_label.loc[x, 'S_INFO_INDUSTRYCODE'])
        stock_concept['股票简称'] = stock_concept['S_CON_WINDCODE'].apply(lambda x: stock_name.loc[x, 'S_INFO_NAME'])
        stock_concept = stock_concept[~stock_concept['Name'].isna()]
        stock_concept = stock_concept[~stock_concept['股票简称'].isna()]
        stock_concept = stock_concept[(stock_concept['Ind_Code'].apply(lambda x: str(x)[:2]) == '02')]  # 02代表概念板块
        concept_alter = stock_concept[(stock_concept['S_CON_OUTDATE'] > str(start_date)) | (stock_concept['S_CON_OUTDATE'].isna())]

        concept_alter['Name'] = concept_alter['Name'].apply(lambda x: np.nan if ((x[-4:] == '(退市)') or (x[-4:] == '金股指数')
                                                                                 or ('国资' in x)  or ('陆股通' in x) or ('精选' in x)) else x)
        concept_alter = concept_alter[~stock_concept['Name'].isna()]
        #######先获取我们筛选后的概念##############
        concet_list = ['高振幅指数','MSCI大盘指数','QFII重仓指数','即将解禁指数','世界杯指数','最小市值指数','全A(非陆股通重仓前100)指数','下跌点位贡献指数',
                       '摘帽指数','打板指数','大央企重组指数','私募重仓指数','债转股指数','举牌指数','高瓴资本指数','领涨龙头指数','首板指数','高价股指数','龙虎榜指数',
                       '证金指数'] # 应该剔除哪些概念

        concept_alter = concept_alter[~(concept_alter['Name'].isin(concet_list))]
        concept_alter['S_CON_OUTDATE'] = concept_alter['S_CON_OUTDATE'].apply(lambda x: int(x) if type(x) == str else x)
        concept_alter['S_CON_INDATE'] = concept_alter['S_CON_INDATE'].astype(int)
        concept_alter['S_CON_WINDCODE'] = concept_alter['S_CON_WINDCODE'].apply(lambda x: stockList.trans_windcode2int(x))
        Concept_list = sorted(list(set(concept_alter['F_INFO_WINDCODE'])))
        stock_list = sorted(list(set(concept_alter['S_CON_WINDCODE'])))

        self.Concept_list = Concept_list
        self.concept_alter = concept_alter
        self.stock_list = stock_list

    def Cal_stock_data(self, save_path=base_address+'FunctionData/'):
        ##########################################1、全市场个股日间数据：###############################################
        self.save_path = save_path
        ##（1)全市场个股每日的涨停价：不会选取ST个股，所以涨停价不会存在5%的情况##
        Limit_price = getData.get_daily_1factor('limit_up_price', date_list=self.date_list)[self.pre_close.columns].astype(float)
        Lowest_Price = getData.get_daily_1factor('limit_down_price', date_list=self.date_list)[self.pre_close.columns].astype(float)

        ##（2）全市场个股每日是否涨停
        code_list = set(self.close.columns).intersection(set(Limit_price.columns))
        Limit_stock = (self.close[code_list] == Limit_price[code_list])

        ##（3）获取全市场可交易的股票池
        # 全市场未开板新股：由于未上市前的状态是NaN，因此可以用上市后累计涨停状态是1 & 最高价=最低价 表示上市后未开板新股；ipo_date=1表示上剔除上市第一日#
        ipo_one_board = self.ipo_date.copy()
        ipo_one_board[ipo_one_board == 1] = 0
        ipo_one_board.replace(0, np.nan, inplace=True)  # 把未上市之前的日期都变为0
        ipo_one_board[ipo_one_board > 0] = 1  # 上市之后的时间都标记为1
        ipo_one_board = (((ipo_one_board * Limit_stock).cumprod() == 1) & (self.high == self.low)) | (
                    self.ipo_date == 1)
        stock_pool = stockList.clean_stock_list(no_ST=True, least_live_days=1, no_pause=True, least_recover_days=0).loc[
                     self.start_date:self.end_date]
        stock_pool = (ipo_one_board == 0) & stock_pool

        ##再用股票池赋值一下##
        Limit_stock = Limit_stock & stock_pool

        ##（5）全市场个股每日是否炸板：收盘价<涨停价 & 最高价=涨停价
        Open_Board_stock = ((self.close[code_list] < Limit_price[code_list]) & (
                    self.high[code_list] == Limit_price[code_list])) & stock_pool

        ##(6)全市场个股连板高度
        limit_up_new = Limit_stock * stock_pool.replace(False, np.nan)
        Limit_High = limit_up_new.cumsum() - limit_up_new.cumsum()[limit_up_new == 0].ffill().fillna(0)
        Limit_High.fillna(0, inplace=True)

        ##（7）全市场日间强势个股
        # 个股60日线+个股的每日涨跌幅
        close_10 = self.close_adj.rolling(10, min_periods=1).mean()
        close_20 = self.close_adj.rolling(20, min_periods=1).mean()
        close_60 = self.close_adj.rolling(60, min_periods=1).mean()
        close_120 = self.close_adj.rolling(120, min_periods=90).mean()
        close_pct = self.close / self.pre_close - 1
        Max_20 = self.close_adj.rolling(20, min_periods=1).max()

        # 4天2板，6天3板，10天4板；3天涨15% & 5天超额15%，10天涨25%，10天涨35%；如果当日未涨停，当日换手率位于市场前20%，且最大回撤不能超过20%
        # 且个股至少大于60日均线
        Power_stock_in = ((Limit_stock.rolling(4).sum() >= 2) & (
                    self.close_adj.pct_change(4).rank(axis=1, ascending=False) <= 100)) | \
                         ((Limit_stock.rolling(6).sum() >= 3) & (
                                     self.close_adj.pct_change(6).rank(axis=1, ascending=False) <= 100)) | \
                         ((Limit_stock.rolling(10).sum() >= 3) & (
                                     self.close_adj.pct_change(10).rank(axis=1, ascending=False) <= 100)) | \
                         ((self.close_adj.pct_change(3) >= 0.15) & ((self.close_adj.pct_change(3).T - self.Index_close[
                             'SZZZ'].pct_change(3)).T >= 0.1) & (
                                      self.close_adj.pct_change(3).rank(axis=1, ascending=False) <= 100)) | \
                         ((self.close_adj.pct_change(5) >= 0.25) & ((self.close_adj.pct_change(5).T - self.Index_close[
                             'SZZZ'].pct_change(5)).T >= 0.15) & (
                                      self.close_adj.pct_change(5).rank(axis=1, ascending=False) <= 100)) | \
                         ((self.close_adj.pct_change(10) >= 0.35) & ((self.close_adj.pct_change(10).T -
                                                                      self.Index_close['SZZZ'].pct_change(
                                                                          10)).T >= 0.2) & (
                                      self.close_adj.pct_change(10).rank(axis=1, ascending=False) <= 100))
        Power_stock_in = Power_stock_in & (Limit_stock | (self.turn.rank(axis=1, pct=True) >= 0.8)) & (
                    self.close_adj / self.close_adj.rolling(20).max() - 1 > -0.15) \
                         & (self.close_adj > close_60) & stock_pool
        # 入选基准日的周期最大值
        Max_close = self.close_adj.fillna(method='ffill').rolling(10, min_periods=1).max()
        Max_close = Max_close[Power_stock_in]
        Max_close.fillna(method='ffill', inplace=True)

        # 写入强势股 #
        Power_stock = pd.DataFrame(False, index=Power_stock_in.index, columns=Power_stock_in.columns)
        Before_Power_in = pd.DataFrame(False, index=Power_stock_in.index, columns=Power_stock_in.columns)
        for date in tqdm(Power_stock.index):
            #######如果今天的强势股=昨天的强势股个股 | 今天新进入池子的
            if date != Power_stock.index[0]:
                yesterday = self.date_list[self.date_list.index(date) - 1]  # 获取昨天的强势股
                power_list = Power_stock.loc[yesterday][Power_stock.loc[yesterday] == True].index.to_list()
                if len(power_list) > 0:
                    #  判断今天是否满足：个股距离入选周期的最高点跌幅大于15%，那就剔除出强势股，入选前期强势股
                    power_choice = (
                                (self.close_adj.loc[date, power_list] / Max_close.loc[date, power_list] - 1) < -0.15)
                    power_del = power_choice[power_choice == True].index.to_list()
                    # 如果个股今日已经连续下跌5日以上，剔除#
                    power_choice = (close_pct.loc[:date, power_list].iloc[-5:] <= 0).sum() == 5
                    power_choice = power_choice[power_choice == True].index.to_list()
                    power_del = list(set(power_del).union(set(power_choice)))
                    # 如果个股在低于60日均线剔除#
                    power_choice = (self.close_adj.loc[date, power_list] < close_60.loc[date, power_list])
                    power_choice = power_choice[power_choice == True].index.to_list()
                    power_del = list(set(power_del).union(set(power_choice)))
                    # 如果个股距离滚动20日的最高点下跌了15%，剔除#
                    power_choice = (self.close_adj.loc[date, power_list] / Max_20.loc[date, power_list]) - 1 < -0.15
                    power_choice = power_choice[power_choice == True].index.to_list()
                    power_del = list(set(power_del).union(set(power_choice)))
                    ##剔除该股票，入选今日的前期强势股；把剩余股票入选今日强势股##
                    Before_Power_in.loc[date, power_del] = True
                    power_list = list(set(power_list).difference(set(power_del)))
                    Power_stock.loc[date, power_list] = True
            Power_stock.loc[date] = (Power_stock.loc[date] | Power_stock_in.loc[date])

        # 写入前期强势股 #
        Before_Max_close = self.close_adj.fillna(method='ffill').fillna(method='backfill').rolling(10).max()
        Before_Max_close = Before_Max_close[Before_Power_in == True]
        Before_Max_close.fillna(method='ffill', inplace=True)

        Before_Power = pd.DataFrame(False, index=Power_stock_in.index, columns=Power_stock_in.columns)
        for date in tqdm(Before_Power.index):
            # 如果今天不是最后一天，就写入明天的前期强势股#
            if date != Before_Power.index[0]:
                yesterday = self.date_list[self.date_list.index(date) - 1]  # 获取昨天的强势股
                before_power_list = Before_Power.loc[yesterday][Before_Power.loc[yesterday] == True].index.to_list()
                if len(before_power_list) > 0:
                    #  判断今天是否满足：个股距离入选周期的最高点跌幅大于10%：
                    before_power_choice = ((self.close_adj.loc[yesterday:date, before_power_list] / Max_close.loc[
                                                                                                    yesterday:date,
                                                                                                    before_power_list] - 1) < -0.1).sum() == 2
                    before_power_del = before_power_choice[before_power_choice == True].index.to_list()
                    # 如果个股今日已经10天内下跌8日以上，剔除#
                    before_power_choice = (close_pct.loc[:date, before_power_list].iloc[-10:] <= 0).sum() >= 8
                    before_power_choice = before_power_choice[before_power_choice == True].index.to_list()
                    before_power_del = list(set(before_power_del).union(set(before_power_choice)))
                    # 如果个股在低于120日均线剔除#
                    before_power_choice = (
                                self.close_adj.loc[date, before_power_list] < close_120.loc[date, before_power_list])
                    before_power_choice = before_power_choice[before_power_choice == True].index.to_list()
                    before_power_del = list(set(before_power_del).union(set(before_power_choice)))
                    ##剔除该股票，入选前期强势股；把剩余股票入选明日强势股##
                    before_power_list = list(set(before_power_list).difference(set(before_power_del)))
                    Before_Power.loc[date, before_power_list] = True
            #######如果今天的前期强势股=昨天已经在池子里的 & 今天新进入池子的 & 不能再强势股池子里的
            Before_Power.loc[date] = (Before_Power.loc[date] | Before_Power_in.loc[date]) & (~Power_stock.loc[date])

        #####近端强势股和远端强势股：纳入周期########
        Power_in_time = pd.DataFrame(index=Power_stock_in.index, columns=Power_stock_in.columns)
        Power_in_time[Power_stock_in == True] = 0  # 被选进来的日期为第0天
        Power_in_time.fillna(1, inplace=True)  # 先把NaN填充为1
        Power_in_time = Power_in_time.cumsum() - Power_in_time.cumsum()[Power_in_time == 0].ffill()
        Power_in_time = Power_in_time[Power_stock == True]  # 必须是强势股才考虑入选时间
        Power_in_time.dropna(how='all', axis=1)

        self.Power_stock = Power_stock  # 当前强势股
        self.Power_stock_in = Power_stock_in  # 今日入选的强势股
        self.Before_Power = Before_Power  # 前期强势股
        self.Power_in_time = Power_in_time  # 强势股的入选时间

        ##（8）全市场日间龙头股
        # 市场前3高度板（最低3板）,M天M-2板或M日M-1板，M日M-2板的个股,且必须是放过量的龙头
        Limit_M2 = pd.DataFrame(0, index=Limit_High.index, columns=Limit_High.columns)  ##M天M-2板
        amt_rol = (self.amt / self.amt.rolling(20, min_periods=1).mean()).loc[Limit_M2.index]
        amt_rol = amt_rol[Limit_stock == True]
        amt_rol.fillna(0, inplace=True)
        amt_rol = amt_rol.rolling(10).max()
        for M in range(5, 10):
            Mday_max = Limit_stock.rolling(M).sum()
            Mday_max = Mday_max * (Mday_max >= M - 2) * (amt_rol.rolling(M).max() > 1.5)
            Limit_M2 = pd.concat([Limit_M2, Mday_max]).max(level=0)
        Limit_M1 = pd.DataFrame(0, index=Limit_High.index, columns=Limit_High.columns)  ##M天M-1板
        for M in range(4, 10):
            Mday_max = Limit_stock.rolling(M).sum()
            Mday_max = Mday_max * (Mday_max >= M - 1) * (amt_rol.rolling(M).max() > 1.5)
            Limit_M1 = pd.concat([Limit_M1, Mday_max]).max(level=0)
        dragon_stock = pd.concat([Limit_M1, Limit_M2, Limit_High * (amt_rol.rolling(5).max() > 1.5)]).max(level=0)
        ####筛选市场高度前三板####
        Limit_market = dragon_stock.copy()
        Limit_market[(Limit_market.T - Limit_market.max(axis=1)).T == 0] = 0
        Limit_market[(Limit_market.T - Limit_market.max(axis=1)).T == 0] = 0
        Limit_market = Limit_market.max(axis=1)
        Limit_market = Limit_market.apply(lambda x: max(3, x))
        #####由于龙头最高位龙头下跌后，并不会立马变化：所以对于高度板滚动3日设计###
        result = pd.Series(Limit_market.index).apply(
            lambda x: 4 if ((Limit_market.loc[x] < 5) & (Limit_market.rolling(3, min_periods=1).max().loc[x] >= 5)) else
            Limit_market.loc[x])
        result.index = Limit_market.index

        Limit_market = result.copy()
        Limit_market = Limit_market.apply(lambda x: min(x, 5))

        # Limit_market=Limit_market.rolling(3,min_periods=1).max()
        ####今天的新晋龙头股######
        Market_Dragon_in = ((dragon_stock.T - Limit_market) >= 0).T & (Limit_stock.rolling(2).sum() >= 1)
        # 个股最近10日的最高高度板
        Limit_Time = dragon_stock.rolling(5, min_periods=1).max()
        Limit_Time_Satisfy = ((Limit_Time.T - Limit_market) >= 0).T
        # 入选基准日的周期最大值
        Dragon_Max_close = self.close_adj.fillna(method='ffill').fillna(method='backfill').rolling(10).max()
        Dragon_Max_close = Dragon_Max_close[Market_Dragon_in == True]
        Dragon_Max_close.fillna(method='ffill', inplace=True)
        # 写入龙头股
        Dragon_stock = pd.DataFrame(False, index=Market_Dragon_in.index, columns=Market_Dragon_in.columns)
        Before_Dragon_stock = pd.DataFrame(False, index=Market_Dragon_in.index, columns=Market_Dragon_in.columns)
        for date in tqdm(Dragon_stock.index):
            #######如果今天的强势股=昨天的强势股个股 | 今天新进入池子的
            if date != Dragon_stock.index[0]:
                yesterday = self.date_list[self.date_list.index(date) - 1]  # 获取昨天的龙头股
                dragon_list = Dragon_stock.loc[yesterday][Dragon_stock.loc[yesterday] == True].index.to_list()
                if len(dragon_list) > 0:
                    #  判断今天是否满足：个股距离入选周期的最高点跌幅大于20%，且连续两天满足，那就剔除出龙头股，入选前期龙头股
                    dragon_choice = ((self.close_adj.loc[yesterday:date, dragon_list] / Dragon_Max_close.loc[
                                                                                        yesterday:date,
                                                                                        dragon_list] - 1) < -0.15).sum() == 2
                    dragon_del = dragon_choice[dragon_choice == True].index.to_list()
                    # 如果个股今日已经连续下跌4日以上，剔除#
                    dragon_choice = (close_pct.loc[:date, dragon_list].iloc[-4:] <= 0).sum() == 4
                    dragon_choice = dragon_choice[dragon_choice == True].index.to_list()
                    dragon_del = list(set(dragon_del).union(set(dragon_choice)))
                    # 如果个股连续5天没有一次涨停，剔除#
                    dragon_choice = (Limit_stock.loc[:date, dragon_list].iloc[-5:].sum()) == 0
                    dragon_choice = dragon_choice[dragon_choice == True].index.to_list()
                    dragon_del = list(set(dragon_del).union(set(dragon_choice)))
                    # 如果个股在低于20日均线剔除#
                    dragon_choice = (self.close_adj.loc[date, dragon_list] < close_10.loc[date, dragon_list])
                    dragon_choice = dragon_choice[dragon_choice == True].index.to_list()
                    dragon_del = list(set(dragon_del).union(set(dragon_choice)))
                    # 如果个股距离滚动20日的最高点下跌了15%，剔除#
                    dragon_choice = (
                                (self.close_adj.loc[date, dragon_list] / Max_20.loc[date, dragon_list]) - 1 < -0.15)
                    dragon_choice = dragon_choice[dragon_choice == True].index.to_list()
                    dragon_del = list(set(dragon_del).union(set(dragon_choice)))
                    # 如果个股近5日内最高高度，低于最高的3个高度板，那就删了 #
                    dragon_choice = Limit_Time_Satisfy.loc[date][Limit_Time_Satisfy.loc[date] == True].index.to_list()
                    dragon_choice = list(set(dragon_list).difference(set(dragon_choice)))  # 在列表中，但是不满足条件的
                    dragon_del = list(set(dragon_del).union(set(dragon_choice)))
                    ##剔除该股票，入选今日的前期强势股；把剩余股票入选今日强势股##
                    Before_Dragon_stock.loc[date, dragon_del] = True
                    dragon_list = list(set(dragon_list).difference(set(dragon_del)))
                    Dragon_stock.loc[date, dragon_list] = True
            Dragon_stock.loc[date] = (Dragon_stock.loc[date] | Market_Dragon_in.loc[date])

        # 写入前期龙头股 #
        Before_Dragon_Max_close = self.close_adj.fillna(method='ffill').fillna(method='backfill').rolling(10).max()
        Before_Dragon_Max_close = Before_Dragon_Max_close[Before_Dragon_stock == True]
        Before_Dragon_Max_close.fillna(method='ffill', inplace=True)
        # 写入前期龙头股
        Before_Dragon = pd.DataFrame(False, index=Before_Dragon_stock.index, columns=Before_Dragon_stock.columns)
        for date in tqdm(Before_Dragon.index):
            # 如果今天不是最后一天，就写入明天的前期龙头股#
            if date != Before_Dragon.index[0]:
                yesterday = self.date_list[self.date_list.index(date) - 1]  # 获取昨天的前期龙头股
                before_drgon_list = Before_Dragon.loc[yesterday][Before_Dragon.loc[yesterday] == True].index.to_list()
                if len(before_drgon_list) > 0:
                    # 如果个股距离入选前期龙头股的日期，再次下跌10%，连续2日满足，那就剔除前期龙头股
                    before_dragon_choice = ((self.close_adj.loc[yesterday:date,
                                             before_drgon_list] / Before_Dragon_Max_close.loc[yesterday:date,
                                                                  before_drgon_list] - 1) < -0.1).sum() == 2
                    before_dragon_del = before_dragon_choice[before_dragon_choice == True].index.to_list()
                    # 如果个股今日已经10天内下跌8日以上，剔除#
                    before_dragon_choice = (close_pct.loc[:date, before_drgon_list].iloc[-10:] <= 0).sum() >= 8
                    before_dragon_choice = before_dragon_choice[before_dragon_choice == True].index.to_list()
                    before_dragon_del = list(set(before_dragon_del).union(set(before_dragon_choice)))
                    # 如果个股在低于60日均线剔除#
                    before_dragon_choice = (
                                self.close_adj.loc[date, before_drgon_list] < close_60.loc[date, before_drgon_list])
                    before_dragon_choice = before_dragon_choice[before_dragon_choice == True].index.to_list()
                    before_dragon_del = list(set(before_dragon_del).union(set(before_dragon_choice)))
                    ##剔除该股票，入选前期强势股；把剩余股票入选明日强势股##
                    before_drgon_list = list(set(before_drgon_list).difference(set(before_dragon_del)))
                    Before_Dragon.loc[date, before_drgon_list] = True
            #######如果今天的前期强势股=昨天已经在池子里的 & 今天新进入池子的 & 不能再强势股池子里的
            Before_Dragon.loc[date] = (Before_Dragon.loc[date] | Before_Dragon_stock.loc[date]) & (
                ~Dragon_stock.loc[date])

        #####近端强势股和远端强势股：纳入周期########
        Dragon_in_time = pd.DataFrame(index=Market_Dragon_in.index, columns=Market_Dragon_in.columns)
        Dragon_in_time[Market_Dragon_in == True] = 0  # 被选进来的日期为第0天
        Dragon_in_time.fillna(1, inplace=True)  # 先把NaN填充为1
        Dragon_in_time = Dragon_in_time.cumsum() - Dragon_in_time.cumsum()[Dragon_in_time == 0].ffill()
        Dragon_in_time = Dragon_in_time[Power_stock == True]  # 必须是强势股才考虑入选时间
        Dragon_in_time.dropna(how='all', axis=1)

        self.Dragon_stock = Dragon_stock
        self.Before_Dragon_stock = Before_Dragon_stock
        self.Market_Dragon_in = Market_Dragon_in

        Power_stock.loc[self.start_date:self.end_date].to_hdf(save_path + 'Power_stock.h5', key='Power_stock',
                                                              format='t')  # 强势股
        Power_stock_in.loc[self.start_date:self.end_date].to_hdf(save_path + 'Power_stock_in.h5', key='Power_stock_in',
                                                                 format='t')  # 每日入选的强势股
        Before_Power.loc[self.start_date:self.end_date].to_hdf(save_path + 'Before_Power.h5', key='Before_Power',
                                                               format='t')  # 前期强势股
        Power_in_time.loc[self.start_date:self.end_date].to_hdf(save_path + 'Power_in_time.h5', key='Power_in_time',
                                                                format='t')  # 当前强势股的入池周期

        Dragon_stock.loc[self.start_date:self.end_date].to_hdf(save_path + 'Dragon_Stock.h5', key='Dragon_Stock',
                                                               format='t')  # 龙头股
        Market_Dragon_in.loc[self.start_date:self.end_date].to_hdf(save_path + 'Market_Dragon_in.h5',
                                                                   key='Market_Dragon_in', format='t')  # 龙头股的入选日期
        Before_Dragon.loc[self.start_date:self.end_date].to_hdf(save_path + 'Before_Dragon.h5', key='Before_Dragon',
                                                                format='t')  # 龙头股的入选日期

        stock_pool.loc[self.start_date:self.end_date].to_hdf(save_path + 'stock_pool.h5', key='stock_pool',
                                                             format='t')  # 股票池
        Limit_price.loc[self.start_date:self.end_date].to_hdf(save_path + 'Limit_price.h5', key='Limit_price',
                                                              format='t')  # 个股涨停价格
        Lowest_Price.loc[self.start_date:self.end_date].to_hdf(save_path + 'Lowest_Price.h5', key='Lowest_Price',
                                                               format='t')  # 个股跌停价格
        Limit_stock.loc[self.start_date:self.end_date].to_hdf(save_path + 'Limit_stock.h5', key='Limit_stock',
                                                              format='t')  # 涨停个股
        Open_Board_stock.loc[self.start_date:self.end_date].to_hdf(save_path + 'Open_Board_stock.h5',
                                                                   key='Open_Board_stock', format='t')  # 炸板个股
        Limit_High.loc[self.start_date:self.end_date].to_hdf(save_path + 'Limit_High.h5', key='Limit_High',
                                                             format='t')  # 连板高度

        self.stock_pool = stock_pool.loc[self.start_date:self.end_date]
        self.Limit_price = Limit_price.loc[self.start_date:self.end_date]
        self.Limit_stock = Limit_stock.loc[self.start_date:self.end_date]
        self.Open_Board_stock = Open_Board_stock.loc[self.start_date:self.end_date]
        self.Limit_High = Limit_High.loc[self.start_date:self.end_date]
        self.Power_stock = Power_stock.loc[self.start_date:self.end_date]

    def Cal_Concept_data(self):
        #######概念板块每日涨跌幅和每日收盘价#######
        code_str = 'S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE,S_DQ_PCTCHANGE'
        sql = r"select %s from wind.AIndexWindIndustriesEOD a where a.TRADE_DT >= '%s' and  a.TRADE_DT <= '%s' " %\
              (code_str,str(self.date_list[0]),str(self.date_list[-1]))
        Price_list = pd.read_sql(sql, con)

        Concept_Close = Price_list.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')
        Concept_Close.index = Concept_Close.index.astype(int)
        Concept_Close = Concept_Close.dropna(how='all')

        Concept_Pct = Price_list.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_PCTCHANGE')
        Concept_Pct.index = Concept_Pct.index.astype(int)
        Concept_Pct = Concept_Pct.dropna(how='all')

        #Concept_Close.loc[self.start_date:self.end_date].to_hdf(save_path + 'BasicData/Concept_Close.h5',
        #                                                        key='Concept_Close', format='t')  # 板块日间收益率
        #Concept_Pct.loc[self.start_date:self.end_date].to_hdf(save_path + 'BasicData/Concept_Pct.h5', key='Concept_Pct',
        #                                                      format='t')  # 板块日间收益率

        self.Concept_Close = Concept_Close.loc[self.start_date:self.end_date]
        self.Concept_Pct = Concept_Pct.loc[self.start_date:self.end_date]

        ##先计算一下全市场每日活跃个股：个股涨幅>1%且个股超额涨幅>1%##
        active_stock = (self.close_adj.pct_change(1) > 0.01) & (
                    (self.close_adj.pct_change(1).T - self.Index_close['SZZZ'].pct_change(1)).T > 0.01)
        ##剔除股性差的个股：过去240日涨停次数小于5次，或者过去240天日滚动10日上涨15%次数少于10次
        active_stock = active_stock & (((self.close_adj.pct_change(1) > 0.09).rolling(240).sum() >= 5) | (
                    (self.close_adj.pct_change(10) > 0.15).rolling(240).sum() >= 10))
        ##剔除空头排列的个股
        del_stock = (self.close_adj < self.close_adj.rolling(5).mean()) & (
        (self.close_adj.rolling(5).mean() < self.close_adj.rolling(10).mean())) & \
                    ((self.close_adj.rolling(10).mean() < self.close_adj.rolling(20).mean()))
        active_stock = active_stock & (~del_stock)
        ##每日不活跃个股即为每日活跃的个股的互斥集
        unactive_stock = ~active_stock

        self.active_stock = active_stock.loc[self.start_date:self.end_date] | self.Dragon_stock.loc[
                                                                              self.start_date:self.end_date]  # 市场上的活跃个股
        self.unactive_stock = unactive_stock.loc[self.start_date:self.end_date]  # 市场上的不活跃个股

    def Active_Concept_choice(self, save_path=base_address+'FunctionData/'):
        #######概念板块有哪些个股#################
        concept_stock = pd.DataFrame(index=self.date_list[250:], columns=self.Concept_list)
        concept_stock = concept_stock.apply(lambda x: [[] for t in x])
        for index in self.concept_alter.index:
            start_time = self.concept_alter.loc[index, 'S_CON_INDATE']
            end_time = self.concept_alter.loc[index, 'S_CON_OUTDATE']
            concept_id = self.concept_alter.loc[index, 'F_INFO_WINDCODE']
            concept_stock.loc[start_time:end_time, concept_id].apply(
                lambda x: x.append(self.concept_alter.loc[index, 'S_CON_WINDCODE']))
        concept_stock = concept_stock.apply(lambda x: x.apply(lambda y: np.nan if y == [] else y))
        concept_stocknum_count = concept_stock.apply(lambda x: x.apply(lambda y: len(y) if type(y) != float else 0))

        Index_Pct = (getData.get_daily_1factor('close', type='bench')[['SZZZ', 'CYBZ']].pct_change(1) * 100).loc[
                    self.start_date:self.end_date]
        self.Index_Pct = Index_Pct['CYBZ']

        '''
        ###每日选出来的活跃板块、活跃概念板块的每日活跃个股、活跃概念板块的每日不活跃个股#
        ############筛选每日活跃概念############
        def Cal_Active_Concept(concept_level, Index_Pct=self.Index_Pct, Concept_Pct=self.Concept_Pct,
                               concept_stocknum_count=concept_stocknum_count,
                               concept_stock=concept_stock, Stock_Pct=self.Stock_Pct, stock_pool=self.stock_pool,
                               Limit_stock=self.Limit_stock, date_list=self.date_list):
            if concept_level == 'small':
                concept_list_small = (concept_stocknum_count >= 5) & (concept_stocknum_count <= 10)
                ####获取概念板块的每日涨幅###
                concept_list_small = concept_list_small & (Concept_Pct.loc[concept_list_small.index] >= 3.5)
                concept_list_small = concept_list_small & ((Concept_Pct.T - Index_Pct).T >= 2)
                ####获取概念板块的个股涨跌幅###
                for date in tqdm(concept_list_small.index):
                    for concept in concept_list_small.columns:
                        if concept_list_small.loc[date, concept] == True:
                            concept_stock_today = concept_stock.loc[date, concept]
                            concept_stock_today = list(set(concept_stock_today).intersection(
                                set(stock_pool.loc[date][stock_pool.loc[date] == True].index.to_list())))
                            ######获取当天该概念:去掉ST，去掉全部个股的涨跌幅###########
                            stock_pct_today = (Stock_Pct.loc[date, concept_stock_today])
                            if (((stock_pct_today >= 5).sum() < 2) | (stock_pct_today.max() <= 9)):
                                concept_list_small.loc[date, concept] = False
                            #######板块内近20天累计最高一板不放进来########
                            else:
                                stock_pct_today = (
                                Limit_stock.loc[date_list[date_list.index(date) - 20]:date, concept_stock_today])
                                if (stock_pct_today).sum().sum() <= 1:
                                    concept_list_small.loc[date, concept] = False
                Index_rank = Concept_Pct[concept_list_small == True].rank(ascending=False, axis=1)
                concept_list_small = (Index_rank <= 3)
                return concept_list_small

            elif concept_level == 'middle':
                concept_list_middle = (concept_stocknum_count >= 11) & (concept_stocknum_count <= 30)
                ####获取概念板块的每日涨幅###
                concept_list_middle = concept_list_middle & (Concept_Pct.loc[concept_list_middle.index] >= 2)
                concept_list_middle = concept_list_middle & ((Concept_Pct.T - Index_Pct).T >= 0.5)
                ####获取概念板块的个股涨跌幅###
                for date in tqdm(concept_list_middle.index):
                    for concept in concept_list_middle.columns:
                        if concept_list_middle.loc[date, concept] == True:
                            concept_stock_today = concept_stock.loc[date, concept]
                            concept_stock_today = list(set(concept_stock_today).intersection(
                                set(stock_pool.loc[date][stock_pool.loc[date] == True].index.to_list())))
                            ######获取当天该概念全部个股的涨跌幅###########
                            stock_pct_today = (Stock_Pct.loc[date, concept_stock_today])
                            ######获取前30%个股的涨幅均值
                            stock_pct = stock_pct_today.sort_values(ascending=False).iloc[
                                        :int(round(len(stock_pct_today) * 0.3, ))]
                            if ((stock_pct.mean() < 5) | (stock_pct.max() <= 9)):
                                concept_list_middle.loc[date, concept] = False
                            #######板块内近60天累计最高一板不放进来########
                            else:
                                stock_pct_today = (
                                self.Limit_stock.loc[date_list[date_list.index(date) - 20]:date, concept_stock_today])
                                if (stock_pct_today).sum().max() <= 1:
                                    concept_list_middle.loc[date, concept] = False

                Index_rank = Concept_Pct[concept_list_middle == True].rank(ascending=False, axis=1)
                concept_list_middle = (Index_rank <= 3)
                return concept_list_middle

            elif concept_level == 'big':
                concept_list_big = (concept_stocknum_count >= 31)
                ####筛选：获取概念板块的每日涨幅###
                concept_list_big = concept_list_big & (Concept_Pct.loc[concept_list_big.index] >= 1.5)
                concept_list_big = concept_list_big & ((Concept_Pct.T - Index_Pct).T > 0)
                ####筛选：获取概念板块的个股涨跌幅###
                for date in tqdm(concept_list_big.index):
                    for concept in concept_list_big.columns:
                        if concept_list_big.loc[date, concept] == True:
                            concept_stock_today = concept_stock.loc[date, concept]
                            concept_stock_today = list(set(concept_stock_today).intersection(
                                set(stock_pool.loc[date][stock_pool.loc[date] == True].index.to_list())))
                            ######获取当天该概念全部个股的涨跌幅###########
                            stock_pct_today = (Stock_Pct.loc[date, concept_stock_today])
                            ######获取前20%个股的涨幅均值
                            stock_pct = stock_pct_today.sort_values(ascending=False).iloc[
                                        :int(round(len(stock_pct_today) * 0.2, ))]
                            if ((stock_pct.mean() < 4) | ((stock_pct > 9).sum() < 2)):
                                concept_list_big.loc[date, concept] = False
                            #######板块内近20天累计最高一板不放进来########
                            else:
                                stock_pct_today = (
                                Limit_stock.loc[date_list[date_list.index(date) - 20]:date, stock_pct.index])
                                if stock_pct_today.sum().max() <= 1:
                                    concept_list_big.loc[date, concept] = False

                Index_rank = Concept_Pct[concept_list_big == True].rank(ascending=False, axis=1)
                concept_list_big = (Index_rank <= 3)
                return concept_list_big

        concept_list_small = Cal_Active_Concept('small')
        concept_list_middle = Cal_Active_Concept('middle')
        concept_list_big = Cal_Active_Concept('big')

        ##活跃概念板块中的全部个股##
        self.concept_list_small = concept_list_small.loc[self.start_date:self.end_date]
        self.concept_list_middle = concept_list_middle.loc[self.start_date:self.end_date]
        self.concept_list_big = concept_list_big.loc[self.start_date:self.end_date]

        concept_active = concept_list_big | concept_list_middle | concept_list_small  # 获取全部的活跃板块
        concept_stock_all = pd.DataFrame(index=self.date_list, columns=self.stock_list)
        ###提取活跃板块中的活跃个股###
        for date in tqdm(concept_active.index):
            for concept in concept_active.columns:
                if concept_active.loc[date, concept] == True:
                    concept_stock_today = list(set(concept_stock.loc[date, concept]))
                    concept_stock_all.loc[date, concept_stock_today] = True
        concept_stock_all.fillna(False, inplace=True)
        '''
        ##活跃股票：位于活跃板块中 & 当日个股超额涨幅>1% & 当日个股涨跌幅>1%
        active_stock = self.active_stock & self.stock_pool
        unactive_stock = self.unactive_stock & self.stock_pool # & concept_stock_all

        active_stock.loc[self.start_date:self.end_date].astype(bool).to_hdf(save_path + 'Active_Stock.h5',
                                                                            key='Active_Stock',
                                                                            format='t')  # 每日活跃板块的活跃个股
        unactive_stock.loc[self.start_date:self.end_date].astype(bool).to_hdf(save_path + 'UnActive_Stock.h5',
                                                                              key='UnActive_Stock',
                                                                              format='t')  # 每日活跃板块的不活跃个股


if __name__ == '__main__':
    ##########################每日数据更新：保存在一个临时文件中，取近几天##########################
    end_date = int(datetime.datetime.now().strftime('%Y%m%d'))
    end_date = 20220914
    date_list = getData.get_date_range(20130101, end_date)
    start_date,end_date = date_list[0],date_list[-1]
    begin = datetime.datetime.now()
    print('数据更新范围:', str(start_date), '-', str(end_date))
    self = BasicData(start_date=start_date, end_date=end_date)
    print('初始化完成:', datetime.datetime.now() - begin)
    self.Cal_stock_data()
    print('全市场股票数据计算完成:', datetime.datetime.now() - begin)
    self.Cal_Concept_data()
    print('板块数据计算完成', datetime.datetime.now() - begin)
    self.Active_Concept_choice()
    print('活跃板块数据计算完成', datetime.datetime.now() - begin)





