import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import requests,json,datetime,time,sys
from dataApi import getData,tradeDate,stockList
from BasicData.local_path import *
from usefulTools import *

def trade_test(signal, trade_type = 'buy',bench='CYBZ',future_days=10,buy_strategy = False, corr = None):
    signal = deal_signal(signal, del_by_corr=False, corr=None, rol=3).loc[20170101:]
    end_date = signal.index[-1]
    signal = signal[signal == True].index.to_list()
    if trade_type == 'buy':
        buy_result = pd.DataFrame(index=signal, columns=['buy_date', 'up_pct', 'up_days', 'down_pct'])
        for date in signal:
            # 先获取买入点
            if buy_strategy == False:
                real_buy_date = date
            else:
                buy_period = get_date_range(date,min(get_pre_trade_date(date,offset=-20),end_date))
                real_buy_date = np.nan
                for buy_date in buy_period:
                    # 卖出点1：如果当日是上涨的，或者收盘价相对于日内低点上涨幅度＞1%，买入
                    if (self.index_close.loc[buy_date, bench]  / self.index_close.loc[get_pre_trade_date(buy_date), bench] -1 >0) | \
                            (self.index_close.loc[buy_date, bench] / self.index_low.loc[buy_date, bench] - 1 > 0.01):
                        real_buy_date = buy_date
                        break
                    elif type(corr) == pd.core.frame.DataFrame:
                        if (corr.loc[buy_date, 'corr'] > -0.2) & (abs(corr.loc[buy_date, 'sentiment_change']) < 1):
                            real_buy_date = buy_date
                            break
            # 然后看卖出点距离最近20日高点的跌幅，和距离最低点的天数
            if np.isnan(real_buy_date) == False:
                watch_period = get_date_range(real_buy_date,min(get_pre_trade_date(real_buy_date, offset=-future_days), end_date))
                max_up = self.index_close.loc[watch_period, bench].max() / self.index_close.loc[
                    real_buy_date, bench] - 1
                up_days = get_trade_date_interval(self.index_close.loc[watch_period, bench].idxmax(), real_buy_date) + 1
                max_down = self.index_close.loc[watch_period, bench].min() / self.index_close.loc[
                    real_buy_date, bench] - 1
                buy_result.loc[date] = real_buy_date, max_up, up_days, max_down
        buy_result = buy_result.astype(float).round(5)
        print('次数', len(buy_result))
        print('胜率',(buy_result['up_pct'] > 0.03).sum() / len(buy_result),(buy_result['up_pct'] > 0.02).sum() / len(buy_result))
        print('平均涨幅', buy_result['up_pct'].mean(), buy_result['up_pct'].median())
        print('平均跌幅', buy_result['down_pct'].mean(), buy_result['down_pct'].median())
        if buy_result['down_pct'].mean() != 0:
            print('涨跌比', buy_result['up_pct'].mean() / buy_result['down_pct'].mean(),buy_result['up_pct'].median() / (np.nan if buy_result['down_pct'].median()==0 else buy_result['down_pct'].median()))
        print('反弹时间', buy_result['up_days'].mean(), buy_result['up_days'].median())

        return buy_result

    elif trade_type == 'sell':
        sell_result = pd.DataFrame(index=signal, columns=['sell_date', 'down_pct', 'down_days', 'up_pct'])
        for date in signal:
            if buy_strategy == False:
                # 先获取卖出点
                real_sell_date = date
            else:
                sell_period = get_date_range(date,min(get_pre_trade_date(date,offset=-10),end_date))
                real_sell_date = np.nan
                for sell_date in sell_period:
                    # 卖出点1：如果日内最低点相对于开盘价下跌幅度＞1%，或者收盘价相对于日内高点下跌幅度＞1%，卖出
                    if ((self.index_low.loc[sell_date,bench] / self.index_open.loc[sell_date,bench] - 1 <-0.01) | \
                        (self.index_close.loc[sell_date,bench] / self.index_high.loc[sell_date,bench] - 1 <-0.01)) & \
                            (self.index_close.loc[sell_date,bench] / self.index_high.loc[get_pre_trade_date(sell_date),bench] - 1 < 0.01):
                        real_sell_date = sell_date
                        break
                    elif type(corr) == pd.core.frame.DataFrame:
                        if (corr.loc[sell_date,'corr'] > -0.2) & (abs(corr.loc[sell_date,'sentiment_change']) < 1):
                            real_sell_date = sell_date
                            break

            # 然后看卖出点距离最近20日低点的跌幅，和距离最低点的天数
            if np.isnan(real_sell_date) == False:
                watch_period = get_date_range(real_sell_date,min(get_pre_trade_date(real_sell_date, offset=-future_days), end_date))
                max_down = self.index_close.loc[watch_period, bench].min() / self.index_close.loc[real_sell_date, bench] - 1
                down_days = get_trade_date_interval(self.index_close.loc[watch_period, bench].idxmin(),real_sell_date) + 1
                max_up = self.index_close.loc[watch_period, bench].max() / self.index_close.loc[real_sell_date, bench] - 1

                sell_result.loc[date] = real_sell_date, max_down, down_days, max_up

        sell_result = sell_result.astype(float).round(5)
        print('次数', len(sell_result))
        print('胜率', (sell_result['down_pct'] < -0.03).sum() / len(sell_result), (sell_result['down_pct'] < -0.02).sum() / len(sell_result))
        print('平均跌幅', sell_result['down_pct'].mean(), sell_result['down_pct'].median())
        print('平均涨幅', sell_result['up_pct'].mean(), sell_result['up_pct'].median())
        if sell_result['up_pct'].mean() != 0:
            print('涨跌比', sell_result['down_pct'].mean() / sell_result['up_pct'].mean(),sell_result['down_pct'].median() / sell_result['up_pct'].median())
        print('下跌时间', sell_result['down_days'].mean(), sell_result['down_days'].median())
        return sell_result

def get_half_weight(N, half_time):
    neg = 0.5**(1/half_time)

    x = [(1/neg**i) for i in range(1,N+1)]
    return [i/sum(x) for i in x]
# 处理信号源
def deal_signal(signal,del_by_corr=True,corr=None,rol=5):
    new_signal = signal.copy()
    new_signal[ContinuousTrueTime(new_signal) > 1] = False
    new_signal[new_signal.rolling(rol).sum() >= 2] = False

    if del_by_corr == True:
        drop_date_list = []
        true_signal = sorted(new_signal[new_signal == True].index.to_list())
        for i in range(1,len(true_signal)):
            if get_trade_date_interval(true_signal[i],true_signal[i-1]) <=10: #如果相邻两个点间隔10日以内，且相关系数始终保持极端负数
                if corr.loc[true_signal[i-1]:true_signal[i],'corr'].max() <-0.4:
                    drop_date_list.append(true_signal[i])

        new_signal.loc[drop_date_list] = False

    return new_signal
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
# 获取申万二级行业的走势
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

def get_SW2_factor(factor,start_date,end_date):
    date_list = get_date_range(get_pre_trade_date(start_date,offset=480),end_date)
    code_ind = get_ind_con('sw_all',2)
    if '801231.SI' in code_ind:
        code_ind.pop('801231.SI')
    sw_close = getData.get_daily_1factor(factor,date_list=date_list,code_list=list(code_ind.keys()),type='SW')

    return sw_close

# 强势个股 和 强势板块
def get_high_stock(start_date,end_date,bench='CYBZ'):
    # 读取数据
    date_list = getData.get_date_range(get_pre_trade_date(start_date, offset=480), end_date)
    index_close = getData.get_daily_1factor('close', date_list=date_list, type='bench')
    stock_pool = stockList.clean_stock_list(no_ST=True, least_live_days=120, no_limit_up=False, no_limit_down=False,
                                            no_pause=True, least_recover_days=0).loc[start_date:end_date]
    close_badj = getData.get_daily_1factor('close_badj', date_list=date_list).astype(float)
    long_line_stock = (close_badj.rolling(5).mean() > close_badj.rolling(20).mean()) & \
                      (close_badj.rolling(10).mean() > close_badj.rolling(20).mean()) & \
                      (close_badj.rolling(20).mean() > close_badj.rolling(30).mean()) & \
                      ((close_badj.rolling(30).mean() > close_badj.rolling(60).mean()))
    long_line_stock = long_line_stock.rolling(5).sum() == 5   # 均线多头排列
    short_minpct = close_badj / close_badj.rolling(20).max() - 1
    code_position = ts_rank(close_badj, rol_day=252)[stock_pool]  # 个股当前的股价在过去252日的位置
    code_max_pct = (close_badj / close_badj.rolling(252).min() - 1)[stock_pool]  # 股价过去一年内的最大涨幅
    close_position = ((close_badj - close_badj.rolling(252).min()) / (
            close_badj.rolling(252).max() - close_badj.rolling(252).min()))[stock_pool]
    no_low_60days = close_badj.rolling(5).min() > close_badj.rolling(60).min()

    bench_high_position = index_close[bench].rolling(20).apply(lambda x:x.idxmax()).dropna().loc[start_date:end_date]  #获取对应指数最近20日的最高点，看抗跌情况

    code_before_position = code_position.loc[bench_high_position.index]
    code_before_position = code_before_position.reindex(bench_high_position.values)
    code_before_position.index = bench_high_position.index  # 在市场下跌周期，市场下跌时间点的个股情况

    close_before_position = close_position.loc[bench_high_position.index]
    close_before_position = close_before_position.reindex(bench_high_position.values)
    close_before_position.index = bench_high_position.index # 在市场下跌周期，市场下跌时间点的个股情况

    # ① 下跌前在高位，当前仍在高位
    still_high_stock = ((code_position > 0.8) & (code_before_position > 0.8)) | (
                (close_position > 0.8) & (close_before_position > 0.8))
    # ② 当前处于多头排列，或者底部涨跌幅排名较大
    go_up = long_line_stock | ((code_max_pct.rank(pct=True, axis=1) > 0.8) & (code_max_pct > 0.2)) & stock_pool
    # ③ 从最近20日高点的回撤不能过大
    no_maxdown = ((short_minpct / code_max_pct)[code_max_pct > 0.2] > -0.5) & \
                 ((short_minpct.rank(pct=True, axis=1) > 0.5) | ((short_minpct / code_max_pct)[go_up].rank(pct=True, axis=1) > 0.8))

    high_stock = still_high_stock & go_up & no_maxdown & no_low_60days & stock_pool

    return high_stock.loc[start_date:end_date]

def get_high_swind(start_date,end_date,bench='CYBZ'):
    date_list = getData.get_date_range(get_pre_trade_date(start_date, offset=480), end_date)
    index_close = getData.get_daily_1factor('close', date_list=date_list, type='bench')

    code_sw = getData.get_daily_1factor('SW2', date_list=get_date_range(start_date, end_date))
    code_num = code_sw.T.apply(lambda x: x.value_counts()).T
    real_ind = get_useful_ind(ind_type='SW2', date_list=get_date_range(start_date, end_date)) & (code_num > 5)

    sw2_close = get_SW2_factor('close',get_pre_trade_date(start_date, offset=480), end_date).dropna(how='all', axis=1)
    sw_position = ts_rank(sw2_close, rol_day=252)[real_ind]
    sw_max_pct = (sw2_close / sw2_close.rolling(252).min() - 1)[real_ind]
    sw_close_position = ((sw2_close - sw2_close.rolling(252).min()) / (sw2_close.rolling(252).max() - sw2_close.rolling(252).min()))[real_ind]

    long_line_ind = (sw2_close.rolling(5).mean() > sw2_close.rolling(20).mean()) & (sw2_close.rolling(10).mean() > sw2_close.rolling(20).mean()) & \
                    (sw2_close.rolling(20).mean() > sw2_close.rolling(30).mean()) & ((sw2_close.rolling(30).mean() > sw2_close.rolling(60).mean()))
    long_line_ind = (long_line_ind.rolling(5).sum() == 5)[real_ind]
    sw_short_minpct = (sw2_close / sw2_close.rolling(20).max() - 1)[real_ind]
    bench_high_position = index_close[bench].rolling(20).apply(lambda x: x.idxmax()).dropna().loc[start_date:end_date]  # 获取对应指数最近20日的最高点，看抗跌情况

    sw_before_position = sw_position.loc[bench_high_position.index]
    sw_before_position = sw_before_position.reindex(bench_high_position.values)
    sw_before_position.index = bench_high_position.index  # 在市场下跌周期，市场下跌时间点的个股情况

    sw_close_before_position = sw_close_position.loc[bench_high_position.index]
    sw_close_before_position = sw_close_before_position.reindex(bench_high_position.values)
    sw_close_before_position.index = bench_high_position.index  # 在市场下跌周期，市场下跌时间点的个股情况

    sw2_high_ind = ((sw_position > 0.8) & (sw_before_position > 0.8)) | ((sw_close_position > 0.8) & (sw_close_before_position > 0.8))
    sw2_go_up = long_line_ind | ((sw_max_pct.rank(pct=True,axis=1) > 0.8) & (sw_max_pct > 0.1))

    sw2_no_maxdown = ((sw_short_minpct / sw_max_pct)[sw2_go_up ==True] > -0.5) & \
                     ((sw_short_minpct.rank(pct=True,axis=1) > 0.5) | ((sw_short_minpct / sw_max_pct.replace(0, np.nan))[sw2_go_up ==True].rank(pct=True,axis=1) > 0.8))

    high_ind = sw2_high_ind & sw2_go_up & sw2_no_maxdown & real_ind

    return high_ind.loc[start_date:end_date]
# 长下影线个股
def long_below_line(open,close,high,low):
    low_line_long = pd.concat([open,close]).min(level=0) - low
    up_line_long = high - pd.concat([open,close]).max(level=0)

    return (low_line_long / up_line_long >1.1) & (low_line_long/(high-low)>0.4) & (up_line_long/(high-low) <0.3),low_line_long/(high-low)

class Daily_Market_Sentiment(object):
    def __init__(self,start_date,end_date,save_path=base_address):
        self.save_path=save_path
        date_list = getData.get_date_range(get_pre_trade_date(start_date,offset=480), end_date)
        self.start_date=date_list[0]
        self.end_date=date_list[-1]
        self.date_list = date_list

        self.save_start_date = start_date
        self.save_end_date = end_date
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
        ipo_date = getData.get_daily_1factor('live_days', date_list=date_list).dropna(how='all')
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
        index_open = getData.get_daily_1factor('open', type='bench')
        index_high = getData.get_daily_1factor('high', type='bench')
        index_low = getData.get_daily_1factor('low', type='bench')
        index_amt = getData.get_daily_1factor('amt', type='bench')
        self.index_close = index_close # 指数收盘价
        self.index_open = index_open
        self.index_high = index_high
        self.index_low = index_low
        self.index_amt = index_amt


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

        # 申万行业
        sw2_close = get_SW2_factor('close',start_date,end_date)
        sw2_low = get_SW2_factor('low', start_date, end_date)
        sw2_high = get_SW2_factor('high', start_date, end_date)
        sw2_open = get_SW2_factor('open', start_date, end_date)
        sw2_amt = get_SW2_factor('amt', start_date, end_date)
        self.sw2_low = sw2_low
        self.sw2_high = sw2_high
        self.sw2_open = sw2_open
        self.sw2_close = sw2_close
        self.sw2_amt = sw2_amt

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
        limit_num = self.highest_stock.loc[start_date:end_date].sum(axis=1)
        double_limit_num =  (self.highest_stock.rolling(2).sum() == 2).loc[start_date:end_date].sum(axis=1)
        open_limit_ratio = (self.open_board_stock.sum(axis=1) / (self.open_board_stock.sum(axis=1) + self.highest_stock.sum(axis=1))).loc[start_date:end_date]
        limit_high_degree = (self.limit_high.rolling(3).max().shift(1) * self.highest_stock + self.highest_stock).max(axis=1).loc[start_date:end_date]
        limit_down_num = self.lowest_stock.loc[start_date:end_date].sum(axis=1)

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
        self.whole_sentiment = whole_sentiment.loc[self.save_start_date:self.save_end_date]

    # 5、汇总异常情况1：情绪处于极端高位或者极端低位
    def sentiment_extremely_position(self,start_date,end_date,rol=5):
        sentiment = self.whole_sentiment.copy()
        sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']] = sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']].rolling(rol).mean()
        # 异常点1：情绪处于极端位置
        buy_sentiment = (sentiment['情绪得分'] <= 2.5) # 情绪处于冰点

        sell_sentiment = (sentiment['情绪得分'] >= 7.5) # 情绪处于高点，但是下跌的高点要剔除情绪从冰点直接拉升到高点的情形
        sell_index = sell_sentiment[sell_sentiment == True].index.to_list()
        for date in sell_index:
            # 如果过去20日有冰点，
            if buy_sentiment.loc[get_pre_trade_date(date,offset=20):date].sum() >=1:
                # 如果市场是连续反弹至高点的，那么该高点是一个非常优秀的高点
                ice_date = buy_sentiment.loc[get_pre_trade_date(date, offset=20):date][buy_sentiment.loc[get_pre_trade_date(date,offset=20):date] == True].index[-1]
                if sentiment.loc[ice_date:date, '情绪得分'].diff().min() > 0:
                    sell_sentiment.loc[date] = 0
                    continue_high_date = get_pre_trade_date(date,-1)
                    while continue_high_date in sell_index:
                        min_date = sentiment.loc[:continue_high_date].iloc[-5:]['情绪得分'].idxmin()
                        max_date = sentiment.loc[:min_date].iloc[-10:]['情绪得分'].idxmax()
                        if (sentiment.loc[max_date,'情绪得分'] - sentiment.loc[min_date,'情绪得分'] > 1) & (sentiment.loc[min_date,'情绪得分'] < 8) & \
                        (sentiment.loc[continue_high_date,'情绪得分']- sentiment.loc[min_date,'情绪得分'] > 0.5):
                            break
                        # 从高位回落（没有回落到7.5以下；再往上的时候仍然算高点）
                        sell_sentiment.loc[continue_high_date] = 0
                        continue_high_date = get_pre_trade_date(continue_high_date, -1)
        extremely_position = buy_sentiment * 1 + sell_sentiment * -1

        return extremely_position.loc[start_date:end_date]
    # 6、汇总异常情况2：情绪和指数发生背离
    def sentiment_derviately(self,start_date,end_date,bench='CYBZ'):
        windows = 10
        corr_line, no_corr_line = -0.4, 0.4
        bench_line,big_bench_line = 0.01, 0.02
        change_line, bigchange_line = 1.5, 2.5
        rol = 5

        sentiment = self.whole_sentiment.copy()
        sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']] = \
            sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']].rolling(rol).mean()

        # 第一步：首先相关系数一定是要出现背离的，所以先关注相关系数的背离:寻找最近10天-5天的区间内，相关系数最低的一天作为背离情况的划分
        corr = pd.DataFrame(index=sentiment.index,columns=['corr','sentiment_change','bench_pct'])
        for date in corr.index[windows:]:
            date_corr = sentiment.loc[get_pre_trade_date(date,offset=windows):date, ['情绪得分', bench]][::-1].expanding().corr()
            date_corr = date_corr.reset_index()[date_corr.reset_index()['level_1'] == bench][['level_0', '情绪得分']].set_index('level_0')
            corr.loc[date,'corr'] = date_corr.iloc[6:11]['情绪得分'].min()
            # 情绪得分的改变幅度
            min_date = date_corr.iloc[6:11]['情绪得分'].idxmin()
            corr.loc[date,'sentiment_change'] = sentiment.loc[date,'情绪得分'] - sentiment.loc[min_date,'情绪得分']
            corr.loc[date,'bench_pct'] = sentiment.loc[date,bench] / sentiment.loc[min_date,bench] -1

        corr = corr.astype(float).round(5).dropna(how='all')
        corr['sentiment'] = sentiment['情绪得分']
        # 背离1：情绪先于指数走弱（刻画：相关系数＜-0.4，情绪下降幅度＞-1.5，指数涨跌幅＞-0.01）
        corr_down_signal = (corr['corr'] <= corr_line) & (corr['sentiment_change'] <= -change_line) & (corr['sentiment'] < 6) & \
                           (((corr['bench_pct'] > - bench_line) & ((corr['bench_pct']>0).rolling(5).sum() >= 3)) | (corr['bench_pct'] >  big_bench_line))
        # 背离2；情绪先于指数走强（刻画：相关系数＜-0.4，情绪上升幅度＞1.5，指数涨跌幅<0.01）
        # 背离2.1：此时情绪并非在高位，则意味着后面要上涨
        corr_up_sginal = (corr['corr'] <= corr_line) & (corr['sentiment_change'] >= change_line) & (corr['sentiment'] < 6) & \
                         (((corr['bench_pct'] < bench_line) & ((corr['bench_pct'] < 0).rolling(5).sum() >= 3)) | (corr['bench_pct'] < -big_bench_line))
        # 背离2.2：如果此时情绪已经在高位了，那么意味着后面要回落
        corr_up_down_signal = (corr['corr'] <= corr_line) & (corr['sentiment_change'] >= change_line) & (corr['sentiment'] >= 6) & \
                         (((corr['bench_pct'] < bench_line) & ((corr['bench_pct'] < 0).rolling(5).sum() >= 3)) | (corr['bench_pct'] < -big_bench_line))
        # corr_up_sginal这个位置有一种情况处理：即如果先触发了corr_up_down_signal卖出信号，则后面触发的corr_up_sginal买入信号无效
        for date in corr_up_sginal[corr_up_sginal == True].index:
            if corr_up_down_signal.loc[get_pre_trade_date(date)] == True:
                del_date = date
                while corr_up_sginal.loc[del_date] == True:
                    corr_up_sginal.loc[del_date] = False
                    del_date = get_pre_trade_date(del_date,-1)
        # corr_up_down_signal这个位置要处理两种情况，；第二种是背离现象先出现，但情绪继续反弹到高位
        for date in corr_up_down_signal[corr_up_down_signal==True].index:
            # 第一种是直接从冰点连续反弹到高位的，那么这个背离不影响
            if corr_up_sginal.loc[get_pre_trade_date(date)] == True:
                del_date = date
                while corr_up_down_signal.loc[del_date] == True:
                    corr_up_down_signal.loc[del_date] = False
                    del_date = get_pre_trade_date(del_date,-1)

            elif corr.loc[get_pre_trade_date(date,10):date,'sentiment'].min() < 2.5:
                ice_date = corr.loc[get_pre_trade_date(date,10):date,'sentiment'].idxmin()
                if (corr.loc[ice_date:date,'sentiment'].diff().iloc[1:] <0).sum() == 0:
                    del_date = date
                    while corr_up_down_signal.loc[del_date] == True:
                        corr_up_down_signal.loc[del_date] = False
                        del_date = get_pre_trade_date(del_date, -1)

        # 背离3：如果情绪和指数的相关系数较低（刻画：abs(相关系数)＜0.2）
        # 背离3.1：如果情绪下降幅度＞-2.5，且指数涨跌幅＞-0.01，则意味着后面会出现下跌
        no_corr_down_signal = (abs(corr['corr']) <= no_corr_line) & (corr['sentiment_change'] <= -bigchange_line) & \
                              (corr['bench_pct'] > - bench_line) & \
                              (self.index_close.loc[start_date:end_date,bench].pct_change(fill_method=None).rolling(3).min() > - 0.02) & \
                              (self.index_close.loc[start_date:end_date,bench].rolling(20).max() > self.index_close.loc[start_date:end_date,bench])

        # 背离3.2：如果情绪上涨幅度＞2.5，且指数涨跌幅＜0.01，则意味着后面会出现上涨
        no_corr_up_signal = (abs(corr['corr']) <= no_corr_line) & (corr['sentiment_change'] >= bigchange_line) & \
                            (corr['bench_pct'] < bench_line) & \
                            (self.index_close.loc[start_date:end_date,bench].pct_change(fill_method=None).rolling(3).max() < 0.02) & \
                            (self.index_close.loc[start_date:end_date,bench].rolling(20).min() < self.index_close.loc[start_date:end_date,bench])

        # 进行交易时间点的汇总
        corr_signal = pd.concat([corr_down_signal.rename('corr_sentiment_down'),
                                 corr_up_sginal.rename('corr_sentiment_up'),corr_up_down_signal.rename('corr_sentiment_up_down'),
                                 no_corr_down_signal.rename('no_corr_down'),no_corr_up_signal.rename('no_corr_up')],axis=1).sort_index().astype(float)

        corr_signal[['corr_sentiment_down','corr_sentiment_up_down','no_corr_down']] = corr_signal[['corr_sentiment_down','corr_sentiment_up_down','no_corr_down']] * -1
        corr_signal[['corr_sentiment_up', 'no_corr_up']] = corr_signal[['corr_sentiment_up', 'no_corr_up']] * 1

        return corr_signal.loc[start_date:end_date],corr

    # （3）基于拐点的策略设计：
    def strategy_design(self,start_date,end_date,bench='CYBZ'):
        # 获取市场高位板块和高位个股
        high_stock, high_ind = get_high_stock(start_date, end_date), get_high_swind(start_date, end_date)
        # 获取情绪冰点
        sentiment_extremely = self.sentiment_extremely_position(start_date, end_date, rol=5).loc[start_date:end_date]
        short_sentiment_extremely = self.sentiment_extremely_position(start_date, end_date, rol=1).loc[start_date:end_date]
        left_sentiment_extremely = (sentiment_extremely == 1) & ((short_sentiment_extremely.shift(1) == 1) | (self.whole_sentiment['情绪得分'] <=1))
        # 获取情绪冰点
        ice_position = deal_signal(left_sentiment_extremely == 1, del_by_corr=False, rol=3)
        ice_position = ice_position[ice_position == True]
        # 需要剔除：排除放量大跌的个股
        big_amt_down = ((self.amt / self.amt.shift(1) - 1 > 1.5) & (self.pct_chg < -5))
        # 策略1：情绪冰点到来，左侧买什么
        result = pd.DataFrame(index=ice_position.index,columns=['num', 'pct', 'pct_median','wl_ratio','wl_median_ratio', 'win_ratio', 'bench_pct', 'bench_win_ratio'])
        sw_result = pd.DataFrame(index=ice_position.index,columns=['num', 'pct', 'pct_median', 'wl_ratio','wl_median_ratio', 'win_ratio', 'bench_pct', 'bench_win_ratio'])
        offset = 10
        for date in ice_position.index:
            ice_end = (sentiment_extremely.loc[date:get_pre_trade_date(date, offset=-offset)].cumsum() > 0) & (sentiment_extremely.loc[date:get_pre_trade_date(date, offset=-offset)] == True)
            ice_end = ice_end[ice_end == True].index[-1]
            # 抗跌个股的选取
            choice_stock = high_stock.loc[date]  & (~big_amt_down.loc[date])

            buy_stock = choice_stock & (self.highest_stock.loc[date] == False)
            buy_stock = buy_stock[buy_stock == True].index.to_list()
            # 先看指数的最大上涨幅度，和如果涨幅过低，那么到截止日期的幅度
            bench_net_value = self.index_close.loc[date:get_pre_trade_date(date, offset=-offset), bench] / self.index_close.loc[date, bench]
            bench_max_up = bench_net_value /bench_net_value.cummin() -1

            bench_sell_date = get_pre_trade_date(date, offset=-offset) if bench_max_up.max() <= 0.02 else bench_max_up.idxmax()
            bench_pct = self.index_close.loc[bench_sell_date, bench] / self.index_close.loc[date, bench] - 1
            # 买入方式：当天跌停的不买，第二天再买；连续大跌的不买。
            period_close = self.close_badj.loc[date:ice_end,buy_stock][(~self.lowest_stock.loc[date:ice_end,buy_stock]) & (self.lowest_stock.loc[date:ice_end, buy_stock].cumsum() <2)]
            period_close = period_close.dropna(how='all',axis=1)
            period_close = pd.concat([period_close,self.close_badj.loc[get_pre_trade_date(ice_end,-1):bench_sell_date,period_close.columns]]).bfill()
            # 卖出方式：如果个股的反弹能够＞5%，那么就按照高点卖，如果反弹幅度＜ 5%，那么就持仓到指数高点卖
            up_code = (((period_close / period_close.loc[date] - 1) >= 0.05).sum() > 0)[(((period_close / period_close.loc[date] - 1) >= 0.05).sum() > 0) == True]
            maxup_pct = period_close[up_code.index].max() / period_close.loc[date,up_code.index] - 1

            all_pct = period_close.loc[bench_sell_date] / period_close.loc[date] - 1
            all_pct.loc[maxup_pct.index] = maxup_pct

            result.loc[date] = len(buy_stock), all_pct.mean(), all_pct.median(), \
                               -all_pct[all_pct >0].mean() / (np.nan if all_pct[all_pct <0].mean() == 0 else all_pct[all_pct <0].mean()), \
                               -all_pct[all_pct >0].median() / (np.nan if all_pct[all_pct <0].median() == 0 else all_pct[all_pct <0].median()), \
                              (all_pct >= 0.03).sum() / len(all_pct), bench_pct, (all_pct > bench_pct).sum() / len(all_pct)


            # 寻找抗跌的申万二级行业
            choice_ind = high_ind.loc[date][high_ind.loc[date] == True].index.to_list()
            # 卖出方式：如果个股的反弹能够＞5%，那么就按照高点卖，如果反弹幅度＜ 5%，那么就持仓到指数高点卖
            sw_period_close = self.sw2_close.loc[date:bench_sell_date,choice_ind]

            sw_up_code = (((sw_period_close / sw_period_close.loc[date] - 1) >= 0.03).sum() > 0)[
                (((sw_period_close / sw_period_close.loc[date] - 1) >= 0.03).sum() > 0) == True]
            sw_maxup_pct = sw_period_close[sw_up_code.index].max() / sw_period_close.loc[date, sw_up_code.index] - 1

            sw_all_pct = sw_period_close.loc[bench_sell_date] / sw_period_close.loc[date] - 1
            sw_all_pct.loc[sw_up_code.index] = sw_maxup_pct

            sw_result.loc[date] = len(choice_ind),sw_all_pct.mean(),sw_all_pct.median(), \
                                  sw_all_pct[sw_all_pct >0].mean()/(np.nan if sw_all_pct[sw_all_pct <0].mean() == 0 else sw_all_pct[sw_all_pct <0].mean()), \
                                  sw_all_pct[sw_all_pct >0].median()/(np.nan if sw_all_pct[sw_all_pct <0].median() == 0 else sw_all_pct[sw_all_pct <0].median()), \
                                  (sw_all_pct > 0.02).sum() / len(sw_all_pct),bench_pct, (sw_all_pct > bench_pct).sum() / len(sw_all_pct)

        # 获取情绪冰点
        ice_position = deal_signal(sentiment_extremely == 1, del_by_corr=False, rol=3)
        ice_position = ice_position[ice_position == True]
        # 策略2：情绪冰点后，市场出现长下影线时，根据日内方向进行跟投；如果市场直接反弹，则根据反弹强度方向进行跟投;如果均没有，则不参与市场
        index_lowline = long_below_line(self.index_open, self.index_close, self.index_high, self.index_low)[0][['SZZZ','CYB','CYBZ','wind_A','avg_A']].max(axis=1)
        index_amt_weight = self.index_amt[bench] / self.index_amt[bench].rolling(5).mean()

        sw_pct = self.sw2_close.pct_change(fill_method=None)
        code_sw = getData.get_daily_1factor('SW2', date_list=get_date_range(start_date, end_date))
        code_num = code_sw.T.apply(lambda x: x.value_counts()).T
        real_ind = get_useful_ind(ind_type='SW2', date_list=get_date_range(start_date, end_date)) & (code_num > 5)
        real_ind = real_ind.drop('801231.SI',axis=1)
        choice_result = pd.DataFrame(index=ice_position.index,columns=['buy_date','num', 'pct', 'pct_median','wl_ratio','wl_ratio_median', 'win_rate','bench_pct', 'bench_win_ratio'])

        for date in ice_position.index:
            choice_ind = []
            # 第一步：如果当天仍然在冰点，或者昨天是冰点
            wait_date = date
            while (sentiment_extremely.loc[wait_date] == 1) | (sentiment_extremely.loc[get_pre_trade_date(wait_date,2):wait_date].max() == 1):
                # 判断当天有没有出现下影线，即下影线长度＞上影线，且下影线的比例 ＞0.4：
                long_down_line = (index_lowline.loc[wait_date] == True) & (index_amt_weight.loc[wait_date] > 1) & \
                        (self.index_close.loc[wait_date,bench] / self.index_close.loc[get_pre_trade_date(wait_date):wait_date,bench].min() - 1 < 0.02)
                market_up = self.index_close.loc[wait_date, bench] / self.index_close.loc[get_pre_trade_date(wait_date), bench] - 1 > 0.015

                if (long_down_line | market_up):
                    trade_date = wait_date
                    ################ 开始进行下影线买入 #########################
                    today_pct = sw_pct.loc[trade_date][real_ind.loc[trade_date][real_ind.loc[trade_date] == True].index]
                    lineup_best = (today_pct > 0.06) | ((today_pct.rank(pct=True) > 0.9) & (today_pct > 0.02))

                    choice_ind = lineup_best
                    choice_ind = choice_ind[choice_ind == True].index.to_list()

                if len(choice_ind) > 0:
                    # 先看指数的最大上涨幅度，和如果涨幅过低，那么到截止日期的幅度
                    bench_net_value = self.index_close.loc[date:get_pre_trade_date(date, offset=-offset), bench] / self.index_close.loc[date, bench]
                    bench_max_up = bench_net_value / bench_net_value.cummin() - 1
                    bench_sell_date = get_pre_trade_date(date,offset=-offset) if bench_max_up.max() <= 0.02 else bench_max_up.idxmax()
                    bench_pct = self.index_close.loc[bench_sell_date, bench] / self.index_close.loc[date, bench] - 1
                    # 如果板块涨跌幅 ＞ 3%，则用最大涨跌幅；反之，则一直持有到结束
                    sw_period_close = self.sw2_close.loc[date:bench_sell_date, choice_ind]

                    sw_up_code = (((sw_period_close / sw_period_close.loc[date] - 1) >= 0.03).sum() > 0)[(((sw_period_close / sw_period_close.loc[date] - 1) >= 0.03).sum() > 0) == True]
                    sw_maxup_pct = sw_period_close[sw_up_code.index].max() / sw_period_close.loc[date, sw_up_code.index] - 1

                    sw_all_pct = sw_period_close.loc[bench_sell_date] / sw_period_close.loc[date] - 1
                    sw_all_pct.loc[sw_up_code.index] = sw_maxup_pct

                    choice_result.loc[date] = wait_date,len(choice_ind), sw_all_pct.mean(), sw_all_pct.median(), \
                                          sw_all_pct[sw_all_pct >0].mean() / (np.nan if sw_all_pct[sw_all_pct <0].mean() == 0 else sw_all_pct[sw_all_pct <0].mean()), \
                                              sw_all_pct[sw_all_pct >0].median() / (np.nan if sw_all_pct[sw_all_pct <0].median() == 0 else sw_all_pct[sw_all_pct <0].median()), \
                                          (sw_all_pct > 0.03).sum() / len(sw_all_pct), bench_pct, (sw_all_pct > bench_pct).sum() / len(sw_all_pct)

                    break

                else:
                    wait_date = get_pre_trade_date(wait_date,-1)

        writer = pd.ExcelWriter('C:/Users/86181/Desktop/情绪测试数据.xlsx')
        result.astype(float).round(5).to_excel(writer, sheet_name='code')
        sw_result.astype(float).round(5).to_excel(writer, sheet_name='sw')
        choice_result.astype(float).round(5).to_excel(writer, sheet_name='sw_choicetime')
        writer.close()

    ####保存数据#######
    def save_Result(self, start_date, end_date):
        sentiment = self.whole_sentiment.copy()
        sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']] = sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '赚钱效应']].rolling(5).mean()

        sentiment = sentiment.loc[start_date:end_date]
        sentiment.index = sentiment.index.astype(str)

        writer = pd.ExcelWriter(self.save_path + str(end_date) + '市场情绪.xlsx')
        sentiment.loc[str(start_date):str(end_date)].to_excel(writer, sheet_name='日间情绪得分')
        round(self.sentiment.loc[start_date:end_date], 4).to_excel(writer, sheet_name='日间各部分数值')
        writer.close()

        sentiment_position_5day = self.sentiment_extremely_position(start_date,end_date,rol=5).loc[start_date:end_date]

        fig = plt.subplots(figsize=(30, 20))
        ax1 = plt.subplot(2, 1, 1)
        ax1.bar(sentiment.index, abs(sentiment_position_5day),color=['lightblue' if x == 1 else 'salmon' for x in sentiment_position_5day], width=1)


        #ax1.legend(['sentiment_score'])
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

        ax4 = ax1.twinx()
        ax4.plot(sentiment.index, sentiment['情绪得分'],color='black')

        ax3 = plt.subplot(2, 1, 2)
        ax3.plot(sentiment.index, sentiment[['投机情绪得分', '龙头情绪得分', '赚钱效应']].values)

        xticks = list(range(0, len(sentiment.index), 10))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
        xlabels = [sentiment.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
        ax3.set_xticks(xticks)
        ax3.legend(['speculation', 'dragon', 'earning'])
        ax3.set_xticklabels(xlabels, rotation=0, fontsize=10)
        for tl in ax3.get_xticklabels():
            tl.set_rotation(90)
        plt.savefig(self.save_path + str(end_date) + '市场情绪.png')
        plt.show()

'''
save_start_date, save_end_date = 20180101, 20221021
sentiment_position_5day = self.sentiment_extremely_position(start_date,end_date,rol=5).loc[save_start_date:save_end_date]
sentiment_position_1day = self.sentiment_extremely_position(start_date,end_date,rol=1).loc[save_start_date:save_end_date]
buy_sentiment,sell_sentiment = (sentiment_position_5day == 1),(sentiment_position_5day == -1)  # 情绪处于冰点

a = trade_test(buy_sentiment, trade_type = 'buy',bench='CYBZ',buy_strategy = False, corr = None)
b = trade_test(sell_sentiment, trade_type = 'sell',bench='CYBZ',future_days=20,buy_strategy = True, corr = None)

corr_signal,corr = self.sentiment_derviately(start_date,end_date,bench=bench)
corr_signal = corr_signal.loc[save_start_date:save_end_date]
corr_signal.loc[(sentiment_position_5day != 0)[(sentiment_position_5day != 0)].index] = 0


c = trade_test((corr_signal.loc[20180101:20221021,'corr_sentiment_down'] == -1) & (sentiment_position_5day == 0), trade_type = 'sell',bench='CYBZ',future_days=10,buy_strategy = False, corr = corr)
d = trade_test((corr_signal.loc[20180101:20221021,'corr_sentiment_up'] ==1)& (sentiment_position_5day == 0), trade_type = 'buy',bench='CYBZ',future_days=10,buy_strategy = False, corr = corr)
e = trade_test((corr_signal.loc[20180101:20221021,'corr_sentiment_up_down'] == -1) & (sentiment_position_5day == 0), trade_type = 'sell',bench='CYBZ',future_days=10,buy_strategy = False, corr = corr)
f = trade_test((corr_signal.loc[20180101:20221021,'no_corr_up'] == 1) & (sentiment_position_5day == 0), trade_type = 'buy',bench='CYBZ',future_days=10,buy_strategy = False, corr = corr)
g = trade_test((corr_signal.loc[20180101:20221021,'no_corr_down'] == -1) & (sentiment_position_5day == 0), trade_type = 'sell',bench='CYBZ',future_days=10,buy_strategy = False, corr = corr)
'''


'''
start_date, end_date,bench = 20160101, 20221021, 'CYBZ'
save_path = base_address + 'MarketMonitor/'
# start_date,end_date = 20200101, int(datetime.datetime.now().strftime('%Y%m%d'))
self=Daily_Market_Sentiment(start_date,end_date=end_date)
self.cal_sentiment()
self.save_Result(20180101,end_date)   #保存
'''