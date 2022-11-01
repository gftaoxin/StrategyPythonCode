import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import requests,json,datetime,time,sys
from dataApi import getData,tradeDate,stockList
from BasicData.local_path import *
from usefulTools import *
from MarketMonitor.market_sentiment import *
from MarketMonitor.north_funds import *

# 画图
def draw_picture(df_continous, df_point, bench_close, start_date,end_date,sav_path=None):
    df1 = df_continous.loc[start_date:end_date].copy()
    df2 = df_point.loc[start_date:end_date].copy()
    df3 = bench_close.loc[start_date:end_date]

    df1.index = df1.index.astype(str)
    df2.index = df2.index.astype(str)
    df3.index = df3.index.astype(str)

    fig = plt.subplots(figsize=(40, 15))
    ax1 = plt.subplot(1, 1, 1)
    ax1.bar(df1.index, abs(df1), color=['lightblue' if x == 1 else 'salmon' for x in df1], width=1)

    ax1.set_title(str(start_date // 10000), size=28)
    xticks = list(range(0, len(df1.index), 10))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
    xlabels = [df1.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=0, fontsize=20)
    for tl in ax1.get_xticklabels():
        tl.set_rotation(90)

    ax2 = ax1.twinx()
    ax2.set_yticks([])
    ax2.bar(df2.index, abs(df2), color=['red' if x == 1 else 'green' for x in df2], width=1)

    ax3 = ax1.twinx()
    ax3.plot(df3.index, df3, color='black')

    if sav_path != None:
        plt.savefig(sav_path + 'sentiment.jpg')
    plt.show()

class AllSignal(object):
    def __init__(self,start_date,end_date,bench):
        date_list = getData.get_date_range(start_date, end_date)
        self.start_date, self.end_date =date_list[0], date_list[-1]
        self.date_list = date_list
        self.bench = bench

        index_close = getData.get_daily_1factor('close', type='bench')
        index_open = getData.get_daily_1factor('open', type='bench')
        index_high = getData.get_daily_1factor('high', type='bench')
        index_low = getData.get_daily_1factor('low', type='bench')
        index_amt = getData.get_daily_1factor('amt', type='bench')
        self.index_close = index_close  # 指数收盘价
        self.index_open = index_open
        self.index_high = index_high
        self.index_low = index_low
        self.index_amt = index_amt
    ######################################### 市场情绪信号汇总 #########################################
    def get_sentiment_data(self,start_date,end_date,save_path = 'E:/MarketMonitor/'):
        sentiment = Daily_Market_Sentiment(start_date, end_date=end_date, save_path = save_path)
        sentiment.cal_sentiment()
        sentiment.save_Result(20210101, end_date)  # 保存

        sentiment_position_5day = sentiment.sentiment_extremely_position(start_date, end_date, rol=5).loc[start_date:end_date]
        sentiment_position_1day = sentiment.sentiment_extremely_position(start_date, end_date, rol=1).loc[start_date:end_date]
        corr_signal = sentiment.sentiment_derviately(start_date, end_date, bench=self.bench)[0].loc[start_date:end_date]
        corr_signal.loc[(sentiment_position_5day != 0)[(sentiment_position_5day != 0)].index] = 0
        # 情绪得分
        sentiment_signal = pd.concat([corr_signal[['corr_sentiment_down', 'corr_sentiment_up']].sum(axis=1).rename('sentiment_deviate_up/down'),
                                      corr_signal['corr_sentiment_up_down'].rename('sentiment_deviate_uptoohigh'),
                                      corr_signal[['no_corr_down', 'no_corr_up']].sum(axis=1).rename('sentiment_nocorr_up/down'),
                                      sentiment_position_5day.rename('sentiment_extremely'),
                                      ], axis=1)
        sentiment_signal['all_sentiment'] = sentiment_signal.sum(axis=1)
        sentiment_signal['oneday_sentiment'] = sentiment_position_1day.loc[start_date:end_date]
        # 情绪指标
        sentiment = sentiment.whole_sentiment.copy().loc[start_date:end_date]
        sentiment['情绪得分_1日'] = sentiment['情绪得分'].copy()
        sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']] = sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']].rolling(5).mean()
        sentiment.index = sentiment.index.astype(str)

        return sentiment_signal, sentiment
    ######################################### 北向资金信号汇总 #########################################
    def get_north_data(self,start_date,end_date):
        north = North_Money_Data(start_date, end_date)

        norht_in_signal, market_in_position = north.north_always_in(5, start_date, end_date, self.bench)  # 北向资金连续净流入
        norht_out_signal, market_out_position, market_turnout_position, market_leftout_position = north.north_always_out(5, start_date, end_date, self.bench)  # 北向资金连续净流出
        reverse_signal, reverse_in_result, reverse_out_result = north.north_reverse_transaction(start_date, end_date,self.bench)  # 北向资金的逆势操作
        north_wrong, north_turn_in_position, north_turn_out_position = north.north_change_wrong(start_date, end_date,self.bench)  # 北向资金的纠错
        corr_signal, trade_result = north.north_market_corr(start_date, end_date, self.bench)  # 北向资金和市场的背离

        # 北向资金信号分为两种：# 第一种是连续信号，即区间大概率是看涨还是看跌（连续净流入信号，连续净流出信号
        north_continous_signal = pd.concat([norht_in_signal['buy_signal'].rename('continous_in'),
                                            norht_out_signal['sell_signal'].rename('continous_out'),
                                            norht_out_signal['turn_short_position'].rename('rightmarket_after_continousout'),
                                            norht_out_signal['together_short_position'].rename('left_market_after_continousout'),
                                            north_wrong['turn_position'].rename('continous_inout_turn_error'),
                                            corr_signal['period'].rename('corr_deviate_signal')], axis=1)
        north_continous_signal['period_signal'] = north_continous_signal.sum(axis=1)

        # 第二种是单点信号，即在该点是看涨还是看跌
        north_position_signal = pd.concat([reverse_signal.sum(axis=1).rename('reverse_signal'),
                                           north_wrong['turn_big_in'].rename('big_inout_turn_error')], axis=1)
        north_position_signal['all_signal'] = north_position_signal.sum(axis=1)

        return north_continous_signal, north_position_signal
    ######################################### 所有信号组合汇总 #########################################
    def get_all_consequency(self,):
        sentiment_signal, sentiment = self.get_sentiment_data(self.start_date,self.end_date)
        north_continous_signal, north_position_signal = self.get_north_data(max(self.start_date,20190101),self.end_date)
        bench_close = self.index_close[self.bench].loc[self.start_date:self.end_date]
        # 特殊标注1：北向资金看多，情绪看空时：此时在北向资金净流入期间情绪端不看空
        north_long_sentiment_short = (north_continous_signal['period_signal'] >= 1) & (sentiment_signal['all_sentiment'] == -1)
        sentiment_signal.loc[north_long_sentiment_short[north_long_sentiment_short == True].index, 'all_sentiment'] = 0

        all_continous_signal = north_continous_signal['period_signal'].reindex(self.date_list).fillna(0)
        all_position_signal = sentiment_signal['all_sentiment'] + north_position_signal['all_signal'].reindex(self.date_list).fillna(0)

        all_continous_signal[all_continous_signal > 1] = 1
        all_continous_signal[all_continous_signal < -1] = -1

        all_position_signal[all_position_signal > 1] = 1
        all_position_signal[all_position_signal < -1] = -1


        all_signal = pd.concat([sentiment['情绪得分'],sentiment['情绪得分_1日'],sentiment_signal['sentiment_deviate_up/down'].rename('情绪背离'),
                                sentiment_signal['sentiment_deviate_uptoohigh'].rename('情绪高位背离(空)'),
                                sentiment_signal['sentiment_nocorr_up/down'].rename('情绪异动'),sentiment_signal['sentiment_extremely'].rename('极端情绪'),
                                sentiment_signal['oneday_sentiment'].rename('单日极端情绪'),
                                north_continous_signal[['continous_in', 'continous_out']].sum(axis=1).rename('北水持续流入/出'),
                                north_continous_signal['rightmarket_after_continousout'].rename('北水右侧市场流出(空)'),
                                north_continous_signal['left_market_after_continousout'].rename('北水左侧市场流出'),
                                north_continous_signal['corr_deviate_signal'].rename('北水和市场走势背离'),
                                north_continous_signal['continous_inout_turn_error'].rename('北水持续流入/出纠错'),
                                north_position_signal['reverse_signal'].rename('北水单日背离'),
                                north_position_signal['big_inout_turn_error'].rename('北水大幅流入/出纠错')
                                ],axis=1)

        self.all_continous_signal = all_continous_signal
        self.all_position_signal = all_position_signal
        self.bench_close = bench_close
        self.all_signal = all_signal

        return all_signal
    ######################################### 画图 ###################################################
    def draw_market_picture(self,start_date,end_date,sav_path='C:/Users/86181/Desktop/'):
        draw_picture(self.all_continous_signal, self.all_position_signal, self.bench_close, start_date, end_date, sav_path)


start_date, end_date,bench = 20160101, 20221028, 'CYBZ'
self = AllSignal(start_date,end_date,bench)
all_signal = self.get_all_consequency()
self.draw_market_picture(20220101, 20221231)


