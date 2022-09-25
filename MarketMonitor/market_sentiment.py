import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import requests,json,datetime,time,sys
from dataApi import getData,tradeDate,stockList
from BasicData.local_path import *
from usefulTools import *

# 强势股
def get_strong_stock(start_date,end_date):
    # （1）去掉ST个股：
    stock_list = stockList.clean_stock_list(no_ST=True, least_live_days=100, start_date=start_date, end_date=end_date)
    # （1）均线多头排列：5日线>20日线>30日线>60日线>240日线 & 收盘价>20日线
    date_list = get_date_range(get_pre_trade_date(start_date,offset=240), end_date)
    close_badj = getData.get_daily_1factor('close_badj',date_list=date_list)
    close = getData.get_daily_1factor('close', date_list=date_list)
    pre_close = getData.get_daily_1factor('pre_close',date_list=date_list)
    line5,line20,line60,line240 = close_badj.rolling(5).mean(),close_badj.rolling(20).mean(),\
                                         close_badj.rolling(60).mean(),close_badj.rolling(240).mean()

    BenchMarkStock = ((close_badj>line20) & (line5 > line20) & (line20 > line60) & (line60>line240))
    # （2）涨跌幅：近30日涨跌幅至少位于市场排名前90%，且自身涨跌幅＞30%
    Max_Pct = close_badj.pct_change(30,fill_method=None)
    Max_Pct_Rank = close_badj.pct_change(30,fill_method=None).rank(pct=True,axis=1)
    PctStock = ((Max_Pct_Rank>0.8) & (Max_Pct>0.2))
    # （3）涨停次数：涨停次数
    limit_up_price = getData.get_daily_1factor('limit_up_price', date_list=date_list)

    Limit_Time = (limit_up_price[close.columns] == close).rolling(30).sum() >= 1  # 30日内涨停次数≥3

    strong_stock = ((BenchMarkStock.loc[start_date:end_date] & PctStock.loc[start_date:end_date])|
                    (PctStock.loc[start_date:end_date] & Limit_Time.loc[start_date:end_date]))  \
                   & stock_list.loc[start_date:end_date]

    return strong_stock

class Daily_Market_Sentiment(object):
    def __init__(self,start_date,end_date,save_path=base_address):
        self.save_path=save_path
        date_list = getData.get_date_range(start_date, end_date)
        self.start_date=date_list[0]
        self.end_date=date_list[-1]
        self.date_list = date_list
        # 基础数据
        pre_close = getData.get_daily_1factor('pre_close', date_list=date_list).dropna(how='all', axis=1).astype(float)
        open = getData.get_daily_1factor('open', date_list=date_list)[pre_close.columns].astype(float)
        high = getData.get_daily_1factor('high', date_list=date_list)[pre_close.columns].astype(float)
        low = getData.get_daily_1factor('low', date_list=date_list)[pre_close.columns].astype(float)
        close = getData.get_daily_1factor('close', date_list=date_list)[pre_close.columns].astype(float)
        amt = getData.get_daily_1factor('amt', date_list=date_list)[pre_close.columns].astype(float)
        turn = getData.get_daily_1factor('free_turn', date_list=date_list)[pre_close.columns].astype(float)
        limit_up_price = getData.get_daily_1factor('limit_up_price', date_list=date_list)[pre_close.columns].astype(float)
        limit_down_price = getData.get_daily_1factor('limit_down_price', date_list=date_list)[pre_close.columns].astype(float)
        pct_chg = getData.get_daily_1factor('pct_chg', date_list=date_list)[pre_close.columns].astype(float)

        self.limit_up_price, self.limit_down_price = limit_up_price, limit_down_price
        self.close, self.open, self.high, self.low, self.pre_close = close, open, high, low, pre_close
        self.pct_chg = pct_chg
        self.amt = amt
        self.turn = turn

        close_badj = getData.get_daily_1factor('close_badj', date_list=date_list)[pre_close.columns].astype(float)
        self.close_badj = close_badj

        free_mv = getData.get_daily_1factor('mkt_free_cap', date_list=date_list)[pre_close.columns].astype(float)
        self.free_mv =free_mv # 自由流通市值


        # 1、投机数据：涨停个股，跌停个股，开板个股，连板个股
        highest_stock = (limit_up_price == close)  # 每日涨停个股
        lowest_stock = (limit_down_price == close)  # 每日跌停个股
        open_board_stock = (limit_up_price > close) & (limit_up_price == high)  # 炸板个股

        # 2、获取全市场可以交易的个股股票池，即只剔除未开板新股：由于未上市前的状态是NaN，因此可以用上市后累计涨停状态是1 & 最高价=最低价 表示上市后未开板新股；ipo_date=1表示上剔除上市第一日#
        ipo_date = getData.get_daily_1factor('live_days', date_list=date_list)
        ipo_one_board = ipo_date.copy()
        ipo_one_board[ipo_one_board == 1] = 0
        ipo_one_board.replace(0, np.nan, inplace=True)  # 把未上市之前的日期都变为0
        ipo_one_board[ipo_one_board > 0] = 1  # 上市之后的时间都标记为1
        ipo_one_board = (((ipo_one_board * highest_stock).cumprod() == 1) & (high == low)) | (ipo_date == 1)
        stock_pool = stockList.clean_stock_list(no_ST=True, least_live_days=1, no_pause=True, least_recover_days=0).loc[start_date:end_date]
        stock_pool = (ipo_one_board == 0) & stock_pool
        self.stock_pool = stock_pool

        # 3、连板个股
        limit_up_new = highest_stock * stock_pool.replace(False, np.nan)
        limit_high = limit_up_new.cumsum() - limit_up_new.cumsum()[limit_up_new == 0].ffill().fillna(0)
        limit_high.fillna(0, inplace=True)

        active_pool = self.active_stock_pool()
        self.active_pool = active_pool

        self.highest_stock = highest_stock & active_pool
        self.lowest_stock = lowest_stock & active_pool
        self.open_board_stock = open_board_stock & active_pool
        self.limit_high = limit_high

        index_close = getData.get_daily_1factor('close',type='bench')
        self.index_close = index_close # 指数收盘价

        # 龙头股：近20日涨幅位于市场前100名 & 涨幅＞30% & 距离最高点回撤不能超过50% * 不能是连续跌停（近10日大跌超过-7%以上的日期≥5）
        dragon_stock = (self.close_badj.pct_change(20) > 0.3) & (self.close_badj.pct_change(20).rank(ascending=False,axis=1) <= 50) & \
                       (self.close_badj / self.close_badj.rolling(20).max() -1 < 0.5) & ( (self.pct_chg < -7).rolling(10).sum() <=3) & \
                       ((self.pct_chg < 0).rolling(10).sum() <= 7) & (self.close_badj > self.close_badj.rolling(60).mean())

        self.dragon_stock = dragon_stock & active_pool

        # 强势股
        strong_stock = get_strong_stock(start_date,end_date)
        self.strong_stock = strong_stock & active_pool & (~dragon_stock)

        ###抄底个股：近5日换手率位于市场前30% & 没有触板,且上涨的个股中，上涨最多的30只个股 ；反之亦然
        NoLimit_Stock = (high<limit_up_price) & (turn.rolling(5).mean().rank(axis=1, pct=True) > 0.7)

        buy_higher = ((close / pre_close - 1)[NoLimit_Stock & (close/pre_close-1>0)].rank(axis=1, ascending=False) <= 30)
        buy_lower = ((close / pre_close - 1)[NoLimit_Stock & (close/pre_close-1<0)].rank(axis=1, ascending=False) <= 30)

        self.buy_higher = buy_higher & active_pool
        self.buy_lower = buy_lower & active_pool

        # 拆分2-追高：昨天未涨停，且最高价＞5%
        High_Buy = ((self.high / self.pre_close - 1) >= 0.05) & NoLimit_Stock
        # 拆分3-抄底买入：最低价＜-5%且最高价低于5%
        Lower_buy = ((self.low / self.pre_close - 1) <= -0.05) & ((self.high / self.pre_close - 1) < 0.05) & NoLimit_Stock
        # 拆分4：昨天横盘震荡，价格位于-5%-5%之间
        Stay_buy = ((self.low / self.pre_close - 1) > -0.05) & ((self.high / self.pre_close - 1) < 0.05) & NoLimit_Stock

        self.High_Buy = High_Buy
        self.Lower_buy = Lower_buy
        self.Stay_buy = Stay_buy

    # 市场活跃股的表现：市场资金活跃度比较高的个股的表现
    def active_stock_pool(self):
        ############################# 尝试内容1：市场总成交额70%的个股作为活跃资金的表现 ######################################
        # 1、剔除ST，上市60日以内的个股
        active_stock_pool = self.stock_pool.copy()#stockList.clean_stock_list(no_ST=True, no_pause=True, least_live_days=60,start_date=self.start_date, end_date=self.end_date)

        # 2、剔除空头个股：
        short_stock = (self.close_badj < self.close_badj.rolling(5).mean()) & \
                      (self.close_badj.rolling(5).mean() < self.close_badj.rolling(10).mean()) & \
                      (self.close_badj.rolling(10).mean() < self.close_badj.rolling(20).mean()) & \
                      (self.close_badj.rolling(20).mean() < self.close_badj.rolling(60).mean())

        # 3、剔除成交量过低，且没有波动的个股
        no_amt = ((self.amt.rolling(5).mean() / self.amt.rolling(20).mean() < 0.6)) & (abs(self.pct_chg) <= 3)

        # 4、剔除波动率过低的个股
        low_price = pd.concat([self.pre_close,self.low],axis=0).min(level=0)
        low_volity = (self.high - low_price <=0.05)

        bad_pool = short_stock | no_amt | low_volity

        active_pool = active_stock_pool & (~bad_pool)

        return active_pool
        '''
        # 等权
        #daily_pct = (self.pct_chg[(active_pool).shift(1)].mean(axis=1).rename('active_stock') / 100).loc[test_date:]
        #net_value = (1 + daily_pct).cumprod().ffill()
        #net_value.iloc[0] = 1
        # 市值加权
        daily_pct = (self.pct_chg / 100 * (active_pool * free_mv).div((active_pool * free_mv).sum(axis=1),axis=0).shift(1)).sum(axis=1).loc[test_date:].rename('active_stock')
        net_value = (1 + daily_pct).cumprod().ffill()
        net_value.iloc[0] = 1

        bench_close = getData.get_daily_1factor('close', date_list=self.date_list, code_list=['SZZZ', 'CYBZ', 'wind_A'],
                                                type='bench').loc[test_date:]
        bench_close = bench_close / bench_close.iloc[0]
        bench_pct = bench_close.pct_change(1)

        all_result = pd.concat([bench_close, net_value], axis=1)

        #all_result['excess'] = (1 + daily_pct - bench_pct['wind_A']).cumprod() - 1
        all_result['num'] = active_pool.sum(axis=1).loc[test_date:] / active_pool.sum(axis=1).loc[test_date:].iloc[0]
        all_result.to_excel('E:/active_stock_compare.xlsx')

        return all_result
        '''

    ######################### 市场情绪的刻画 ################################
    # 1、投机氛围：即涨停情况，炸板情况，连板情况，高标特征
    def cal_speculation(self, start_date, end_date, rol_short = 120, rol_long = 480):

        date_list = get_date_range(start_date,end_date)
        # 数据原始值
        speculation_sentiment = pd.DataFrame(index = date_list, columns = ['涨停数量','连板数量','炸板率','连板高度','跌停数量'])
        limit_num = self.highest_stock.loc[date_list].sum(axis=1)
        double_limit_num =  (self.highest_stock.rolling(2).sum() == 2).loc[date_list].sum(axis=1)
        open_limit_ratio = (self.open_board_stock.sum(axis=1) / (self.open_board_stock.sum(axis=1) + self.highest_stock.sum(axis=1))).loc[date_list]
        limit_high_degree = (self.limit_high.rolling(3).max().shift(1) * self.highest_stock + self.highest_stock).max(axis=1).loc[date_list]
        limit_down_num = self.lowest_stock.loc[date_list].sum(axis=1)

        speculation_sentiment['涨停数量'] = limit_num
        speculation_sentiment['连板数量'] = double_limit_num
        speculation_sentiment['炸板率'] = open_limit_ratio
        speculation_sentiment['连板高度'] = limit_high_degree
        speculation_sentiment['跌停数量'] = limit_down_num

        # 数据得分值
        speculation_score = pd.DataFrame(index = date_list, columns = ['涨停数量','连板数量','炸板率','连板高度','跌停数量'])
        limit_score = (ts_rank(limit_num, rol_day=rol_short) + ts_rank(limit_num, rol_day=rol_long))*5
        double_limit_score = (ts_rank(double_limit_num, rol_day=rol_short) + ts_rank(double_limit_num, rol_day=rol_long)) * 5
        open_limit_score = open_limit_ratio.apply(lambda x:10 if x<=0.15 else 8 if ((x>0.15)&(x<=0.2)) else 6 if ((x>0.2)&(x<=0.3)) else 4 if ((x>0.3)&(x<=0.4))else 2 if ((x>0.4)&(x<0.5)) else 0)

        limithigh_num = (self.limit_high.T == limit_high_degree).sum()
        limithigh_score = limit_high_degree.apply(lambda x: 10 if x >= 6 else 8 if x == 5 else 6 if x == 4 else 4 if x == 3 else 2 if x == 2 else 0)
        LimitHigh_add = ((limithigh_num - 1) * 0.5).apply(lambda x: min(2, max(x,0)))
        LimitHigh_add[limit_high_degree <= 2] = 0
        limit_high_score = limithigh_score + LimitHigh_add

        limit_down_score = 10 - (ts_rank(limit_down_num, rol_day=rol_short) + ts_rank(limit_down_num, rol_day=rol_long)) * 5

        speculation_score['涨停数量'] = limit_score
        speculation_score['连板数量'] = double_limit_score
        speculation_score['炸板率'] = open_limit_score
        speculation_score['连板高度'] = limit_high_score
        speculation_score['跌停数量'] = limit_down_score

        return speculation_sentiment, speculation_score

    # 2、龙头情绪：市场中龙头股，和强势股的表现：投机龙头，趋势中军。1日赚钱效应，2日赚钱效应，3日赚钱效应，5日赚钱效应，10日赚钱效应
    def cal_dragon_situation(self,start_date, end_date, rol_short = 120, rol_long = 480):
        date_list = get_date_range(start_date, end_date)
        stock_pct1, stock_pct2 = self.close_badj.pct_change(1), self.close_badj.pct_change(2)
        stock_pct3, stock_pct5 = self.close_badj.pct_change(3), self.close_badj.pct_change(5)

        dragon_sentiment = pd.DataFrame(index=date_list, columns=['龙头股数量', '龙头股溢价率', '强势股数量', '强势股溢价率'])
        ########1、龙头股############
        dragon_pct_1day = stock_pct1[self.dragon_stock.shift(1)].mean(axis=1)
        dragon_pct_2day = stock_pct2[self.dragon_stock.shift(2)].mean(axis=1)
        dragon_pct_3day = stock_pct3[self.dragon_stock.shift(3)].mean(axis=1)
        dragon_pct_5day = stock_pct5[self.dragon_stock.shift(5)].mean(axis=1)


        dragon_sentiment['龙头股数量'] = self.dragon_stock.sum(axis=1)
        dragon_sentiment['龙头股溢价率'] = (dragon_pct_1day + dragon_pct_2day + dragon_pct_3day + dragon_pct_5day) / 4


        ########2、强势股############
        strong_pct_1day = stock_pct1[self.strong_stock.shift(1)].mean(axis=1)
        strong_pct_2day = stock_pct2[self.strong_stock.shift(2)].mean(axis=1)
        strong_pct_3day = stock_pct3[self.strong_stock.shift(3)].mean(axis=1)
        strong_pct_5day = stock_pct5[self.strong_stock.shift(5)].mean(axis=1)

        dragon_sentiment['强势股数量'] = self.strong_stock.sum(axis=1)
        dragon_sentiment['强势股溢价率'] = (strong_pct_1day + strong_pct_2day + strong_pct_3day + strong_pct_5day) / 4

        # 得分
        dragonpct_score = (ts_rank(dragon_sentiment['龙头股溢价率'], rol_day=rol_short) + ts_rank(dragon_sentiment['龙头股溢价率'],rol_day=rol_long)) * 5
        strongpct_score = (ts_rank(dragon_sentiment['强势股溢价率'], rol_day=rol_short) + ts_rank(dragon_sentiment['强势股溢价率'],rol_day=rol_long)) * 5
        dragon_num_score = (ts_rank(dragon_sentiment['龙头股数量'], rol_day=rol_short) + ts_rank(dragon_sentiment['龙头股数量'],rol_day=rol_long)) * 5
        strong_num_score = (ts_rank(dragon_sentiment['强势股数量'], rol_day=rol_short) + ts_rank(dragon_sentiment['强势股数量'],rol_day=rol_long)) * 5

        dragon_score = pd.concat([dragonpct_score, strongpct_score, dragon_num_score, strong_num_score], axis=1)

        return dragon_sentiment, dragon_score

    # 3、赚钱效应：涨停股赚钱效应，追高低吸个股的赚钱效应.1日赚钱效应，2日赚钱效应，3日赚钱效应，5日赚钱效应，10日赚钱效应
    def cal_earning_continuous(self,start_date, end_date,rol_short = 120, rol_long = 480):
        date_list = get_date_range(start_date, end_date)
        # 数据原始值
        earning_sentiment = pd.DataFrame(index=date_list, columns=['涨停溢价率', '炸板溢价率', '追高溢价率', '抄底溢价率'])

        stock_pct1,stock_pct2 = self.close_badj.pct_change(1), self.close_badj.pct_change(2)
        stock_pct3,stock_pct5 = self.close_badj.pct_change(3), self.close_badj.pct_change(5)

        #####1、昨日涨停个股溢价率##########
        limit_pct_1day = stock_pct1[self.highest_stock.shift(1)].mean(axis=1)
        limit_pct_2day = stock_pct2[self.highest_stock.shift(2)].mean(axis=1)
        limit_pct_3day = stock_pct3[self.highest_stock.shift(3)].mean(axis=1)
        limit_pct_5day = stock_pct5[self.highest_stock.shift(5)].mean(axis=1)

        earning_sentiment['涨停溢价率'] = (limit_pct_1day + limit_pct_2day + limit_pct_3day + limit_pct_5day)/4

        ######2、昨日炸板个股溢价率##########
        open_pct_1day = stock_pct1[self.open_board_stock.shift(1)].mean(axis=1)
        open_pct_2day = stock_pct2[self.open_board_stock.shift(2)].mean(axis=1)
        open_pct_3day = stock_pct3[self.open_board_stock.shift(3)].mean(axis=1)
        open_pct_5day = stock_pct5[self.open_board_stock.shift(5)].mean(axis=1)

        earning_sentiment['炸板溢价率'] = (open_pct_1day + open_pct_2day + open_pct_3day + open_pct_5day)/4

        ######3、昨日追高个股溢价率##########
        higher_pct_1day = stock_pct1[self.buy_higher.shift(1)].mean(axis=1)
        higher_pct_2day = stock_pct2[self.buy_higher.shift(2)].mean(axis=1)
        higher_pct_3day = stock_pct3[self.buy_higher.shift(3)].mean(axis=1)
        higher_pct_5day = stock_pct5[self.buy_higher.shift(5)].mean(axis=1)

        earning_sentiment['追高溢价率'] = (higher_pct_1day + higher_pct_2day + higher_pct_3day + higher_pct_5day) / 4

        ########4、昨日抄底板块溢价率########
        lower_pct_1day = stock_pct1[self.buy_lower.shift(1)].mean(axis=1)
        lower_pct_2day = stock_pct2[self.buy_lower.shift(2)].mean(axis=1)
        lower_pct_3day = stock_pct3[self.buy_lower.shift(3)].mean(axis=1)
        lower_pct_5day = stock_pct5[self.buy_lower.shift(5)].mean(axis=1)

        earning_sentiment['抄底溢价率'] = (lower_pct_1day + lower_pct_2day + lower_pct_3day + lower_pct_5day) / 4

        # 数据得分值
        limit_score = (ts_rank(earning_sentiment['涨停溢价率'], rol_day=rol_short) + ts_rank(earning_sentiment['涨停溢价率'], rol_day=rol_long)) * 5
        open_score = (ts_rank(earning_sentiment['炸板溢价率'], rol_day=rol_short) + ts_rank(earning_sentiment['炸板溢价率'], rol_day=rol_long)) * 5
        high_score = (ts_rank(earning_sentiment['追高溢价率'], rol_day=rol_short) + ts_rank(earning_sentiment['追高溢价率'], rol_day=rol_long)) * 5
        low_score = (ts_rank(earning_sentiment['抄底溢价率'], rol_day=rol_short) + ts_rank(earning_sentiment['抄底溢价率'], rol_day=rol_long)) * 5

        earning_score= pd.concat([limit_score, open_score, high_score, low_score],axis=1)

        return earning_sentiment, earning_score

        pass

    # 4、汇总计算得到总的市场情绪：
    def cal_sentiment(self):
        ####1、投机情绪#####
        speculation_sentiment, speculation_score = self.cal_speculation(self.start_date,self.end_date)
        ####2、龙头情绪####
        dragon_sentiment, dragon_score = self.cal_dragon_situation(self.start_date,self.end_date)
        ####3、赚钱效应#####
        earning_sentiment, earning_score = self.cal_earning_continuous(self.start_date,self.end_date)

        sentiment = pd.concat([speculation_sentiment,dragon_sentiment,earning_sentiment],axis=1)
        sentiment_score = pd.concat([speculation_score,dragon_score,earning_score],axis=1)

        self.sentiment = sentiment
        self.sentiment_score = sentiment_score

        #############计算权重#################
        weight = pd.DataFrame(index=sentiment_score.index,columns=sentiment_score.columns)

        weight['涨停数量'], weight['连板数量'], weight['炸板率'], weight['连板高度'], weight['跌停数量'] =\
            0.15, 0.15, 0.05, 0.05,0.1

        weight['龙头股溢价率'],weight['强势股溢价率'] = 0.1, 0.1

        weight['龙头股溢价率'][sentiment['龙头股数量'] <= 10] = 0.05
        weight['强势股溢价率'][sentiment['强势股数量'] <= 30] = 0.05

        weight['涨停溢价率'], weight['炸板溢价率'], weight['追高溢价率'], weight['抄底溢价率'] = \
            0.1, 0.1, 0.05,0.05

        ############获取情绪得分##################
        whole_sentiment = pd.DataFrame(index = sentiment.index ,columns = ['CYBZ','SZZZ','wind_A','情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应'])
        Score = (sentiment_score * weight)

        whole_sentiment[['CYBZ','SZZZ','wind_A']] = self.index_close[['CYBZ','SZZZ','wind_A']]

        whole_sentiment['投机情绪得分']=Score[['涨停数量','连板数量','炸板率','连板高度','跌停数量']].sum(axis=1)/0.5
        whole_sentiment['龙头情绪得分']=Score[['龙头股溢价率','强势股溢价率']].sum(axis=1)/0.2
        whole_sentiment['赚钱效应']=Score[['涨停溢价率','炸板溢价率','追高溢价率','抄底溢价率']].sum(axis=1)/0.3
        whole_sentiment['情绪得分'] = Score.sum(axis=1)

        whole_sentiment = round(whole_sentiment.astype(float),4)
        self.whole_sentiment = whole_sentiment

    ####保存数据#######
    def save_Result(self,start_date,end_date):

        sentiment = self.whole_sentiment.copy()
        sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']] = sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']].rolling(5).mean()
        sentiment = sentiment.loc[start_date:end_date]
        sentiment.index = sentiment.index.astype(str)

        writer = pd.ExcelWriter(self.save_path + str(end_date) + '市场情绪.xlsx')
        sentiment.loc[str(start_date):str(end_date)].to_excel(writer, sheet_name='日间情绪得分')
        round(self.sentiment.loc[start_date:end_date], 4).to_excel(writer, sheet_name='日间各部分数值')
        writer.close()

        fig = plt.subplots(figsize=(30, 20))
        ax1 = plt.subplot(2, 1, 1)

        ax1.plot(sentiment.index, sentiment['情绪得分'])
        ax1.legend(['sentiment_score', 'corr'])
        ax1.set_title('market_sentiment', size=18)
        ax1.set_xticks([])
        xticks = list(range(0, len(sentiment.index), 10))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
        xlabels = [sentiment.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xlabels, rotation=0, fontsize=10)
        for tl in ax1.get_xticklabels():
            tl.set_rotation(90)
        ax2 = ax1.twinx()
        ax2.plot(sentiment.index, sentiment['CYBZ'], color='r')

        ax3 = plt.subplot(2, 1, 2)
        ax3.plot(sentiment.index, sentiment[['投机情绪得分', '龙头情绪得分', '赚钱效应']].values)

        xticks = list(range(0, len(sentiment.index), 10))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
        xlabels = [sentiment.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
        ax3.set_xticks(xticks)
        ax3.legend(['speculation', 'dragon', 'earning'])
        ax3.set_xticklabels(xlabels, rotation=0, fontsize=10)
        for tl in ax3.get_xticklabels():
            tl.set_rotation(90)
        plt.savefig(self.save_path + str(end_date)+'市场情绪.png')
        plt.show()


start_date, end_date = 20160101, 20220919
save_path = base_address + 'MarketMonitor/'
# start_date,end_date = 20200101, int(datetime.datetime.now().strftime('%Y%m%d'))
self=Daily_Market_Sentiment(start_date,end_date=end_date)
self.cal_sentiment()
self.save_Result(20220101,end_date)   #保存
