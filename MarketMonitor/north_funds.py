import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import requests,json,datetime,time,sys
from dataApi import getData,tradeDate,stockList
from BasicData.local_path import *
from usefulTools import *
# 函数1：获取某个日期的前后两日
def get_entanglement_type(price_change,today):
    yesterday = list(price_change.index)[list(price_change.index).index(today) - 1]
    tomorrow = list(price_change.index)[list(price_change.index).index(today) + 1]
    # 顶分型满足：第二根K线的高点是3根K线中的最高点，低点是3根K线中的最高点
    # 底分型满足：第二根K线的高点是3根K线中的最低点，低点是3根K线中的最低点
    Up_Type = (price_change.loc[today, 'high'] > max(price_change.loc[yesterday, 'high'],price_change.loc[tomorrow, 'high'])) & \
              (price_change.loc[today, 'low'] > max(price_change.loc[yesterday, 'low'],price_change.loc[tomorrow, 'low']))
    Down_Type = (price_change.loc[today, 'high'] < min(price_change.loc[yesterday, 'high'],price_change.loc[tomorrow, 'high'])) & \
                (price_change.loc[today, 'low'] < min(price_change.loc[yesterday, 'low'],price_change.loc[tomorrow, 'low']))

    return yesterday,tomorrow,Up_Type,Down_Type


class North_Money_Data(object):
    def __init__(self,start_date,end_date,save_path=base_address):
        self.save_path = save_path
        date_list = getData.get_date_range(start_date, end_date)
        self.start_date, self.end_date =date_list[0], date_list[-1]
        self.date_list = date_list
        # 基础数据
        date_list = get_date_range(start_date, end_date)
        close = get_daily_1factor('close', date_list=date_list).dropna(how='all')
        high = get_daily_1factor('high', date_list=date_list).dropna(how='all')
        low = get_daily_1factor('low', date_list=date_list).dropna(how='all')
        code_ind = get_daily_1factor('SW1', date_list=date_list)
        self.close, self.high, self.low = close, high, low
        ind_name = list(get_ind_con(ind_type='SW', level=1).keys())
        # 指数的基础数据
        bench_close = getData.get_daily_1factor('close', date_list=date_list, type='bench')
        bench_open = getData.get_daily_1factor('open', date_list=date_list, type='bench')
        bench_high = getData.get_daily_1factor('high', date_list=date_list, type='bench')
        bench_low = getData.get_daily_1factor('low', date_list=date_list, type='bench')

        self.bench_close, self.bench_open, self.bench_high, self.bench_low = bench_close, bench_open, bench_high, bench_low

        bench_amt = getData.get_daily_1factor('amt', type='bench',date_list=date_list)
        self.bench_amt = bench_amt

        north_data = getData.get_daily_1factor('north_funds',date_list=date_list) # 北向资金数据
        north_vol_data = getData.get_daily_1factor('north_quantity',date_list=date_list) # 北向资金的个股持有数量
        north_vol_in = north_vol_data.diff(1) # 北向资金的个股净流入数量
        north_amt = (north_vol_in * close).dropna(how='all') # 北向资金的个股净流入金额
        north_ind_amt = pd.concat([north_amt[code_ind == x].sum(axis=1).rename(x) for x in ind_name],axis=1)  # 单日行业的净流入

        self.north_data = north_data
        self.north_vol_data = north_vol_data
        self.north_vol_in = north_vol_in
        self.north_amt = north_amt
        self.north_ind_amt = north_ind_amt

        # 1、北向资金净买入金额占总金额的比例
        north_buy_weight = north_data['当日买入成交净额(人民币)'] / north_data[['当日买入成交金额(人民币)', '当日卖出成交金额(人民币)']].sum(axis=1)
        north_buy_weight = north_buy_weight.dropna()

        self.north_buy_weight = north_buy_weight

        # 2、北向资金的买入个股比例
        north_buy_stock_weight = (north_vol_in > 0).sum(axis=1) / (~np.isnan(north_vol_in)).sum(axis=1)
        north_buy_stock_weight = north_buy_stock_weight.reindex(north_buy_weight.index).dropna()

        self.north_buy_stock_weight = north_buy_stock_weight

        # 3、北向资金净买入行业
        north_buy_ind_weight = (north_ind_amt > 0).sum(axis=1) / (~np.isnan(north_ind_amt)).sum(axis=1)
        north_buy_ind_weight = north_buy_ind_weight.reindex(north_buy_weight.index).dropna()
        self.north_buy_ind_weight = north_buy_ind_weight
        # 北向资金涌入行业比例
        north_ind_amt = north_ind_amt.replace(0, np.nan).dropna(how='all')
        north_ind_amt5days = north_ind_amt.fillna(0).rolling(5).sum()[~np.isnan(north_ind_amt)].dropna(how='all')
        self.north_ind_amt5days = north_ind_amt5days

    # 1、北向资金的连续净流入
    def north_alway_in(self):
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        # 我们在计算累计净流入时，一些小额度的净流出对着你影响非常小
        # 如果当前是累计5日净流入，那么下一日的净流出如果＜20亿，那么仍然算作净流入
        new_north_data_in = new_north_data.copy()
        new_north_data_in[((new_north_data['当日买入成交净额(人民币)']>0).rolling(5).sum() == 5).shift(1) & (new_north_data['当日买入成交净额(人民币)'] > -20)
                       & (new_north_data['当日买入成交净额(人民币)'] < 0)] = 0

        continue_in = pd.Series(index = new_north_data.index)
        i = 0
        for date in new_north_data.index:
            if (new_north_data.loc[date,'当日买入成交净额(人民币)'] > 0) :
                if i <0:
                    i = 1
                else:
                    i += 1
                continue_in.loc[date] = i

            elif i >=5:
                max_money = new_north_data.loc[:date, '当日买入成交净额(人民币)'].iloc[-i - 1:-1].mean()

                if (new_north_data.loc[date, '当日买入成交净额(人民币)'] > -min(max(max_money,20),40)):
                    i += 1
                    continue_in.loc[date] = i
                else:
                    i = -1
                    continue_in.loc[date] = i

            elif i <0:
                i -= 1
                continue_in.loc[date] = i

            else:
                i = -1
                continue_in.loc[date] = i

        continue_money = continue_in.copy()
        #continue_money[continue_money > 5] = 5
        #continue_money[(continue_money < 5)] = 0
        #continue_in = ContinuousTrueTime(new_north_data_in['当日买入成交净额(人民币)'] >= 0)
        #continue_out = ContinuousTrueTime(new_north_data['当日买入成交净额(人民币)'] < 0)



    # 指数缠论拐点
    # 指数缠论拐点
    def Kline_Bench_Daily(self,bench_code):
        # 用缠论的方式构造顶分型，和底分型
        Kline = pd.concat([self.bench_high[bench_code].rename('high'), self.bench_low[bench_code].rename('low')], axis=1)
        # 第一步：处理K线包含关系：默认第一天为起始点，建立第一天的情况
        price_change = Kline.loc[[self.start_date]]
        for now_time in self.date_list[1:]:
            # 先找到上一个索引（上一个索引指的是处理后的K线之间的相互关系），确定两个索引之间是否存在包含关系：
            # 包含关系：两天中，一天的最高价＞另一天的最高价，一天的最低价＜另一天的最低价
            involve = ((Kline.loc[now_time, 'high'] >= price_change.iloc[-1]['high']) & (Kline.loc[now_time, 'low'] <= price_change.iloc[-1]['low'])) | \
                      (Kline.loc[now_time, 'high'] <= price_change.iloc[-1]['high']) & (Kline.loc[now_time, 'low'] >= price_change.iloc[-1]['low'])
            # 如果不存在包含关系，则直接把该K线加入处理后的K线中，并继续进行循环
            if involve == False:
                price_change.loc[now_time] = Kline.loc[now_time]
            else:
                # 如果存在包含关系，则确定是向上还是向下
                # 如果是第一根K线，则直接用第一根K线涨跌幅来确定
                if len(price_change) > 1:
                    up = price_change.iloc[-1]['high'] >price_change.iloc[-2]['high']
                    down =  price_change.iloc[-1]['low'] < price_change.iloc[-2]['low']
                else:
                    start_time = price_change.index[0]
                    up = self.bench_close.loc[start_time,bench_code] >= self.bench_open.loc[start_time,bench_code]
                    down = self.bench_close.loc[start_time,bench_code] < self.bench_open.loc[start_time,bench_code]

                # 如果是上升，那么就取两者high和low的最大值；如果是下降，那么就取两者high和low的最小值
                if up == True:
                    price_change.iloc[-1]['high'] = max( price_change.iloc[-1]['high'],Kline.loc[now_time, 'high'])
                    price_change.iloc[-1]['low'] = max( price_change.iloc[-1]['low'],Kline.loc[now_time, 'low'])

                elif down == True:
                    price_change.iloc[-1]['high'] = min(price_change.iloc[-1]['high'], Kline.loc[now_time, 'high'])
                    price_change.iloc[-1]['low'] = min(price_change.iloc[-1]['low'], Kline.loc[now_time, 'low'])

                # 最后把日期往后挪一个
                price_change = price_change.rename(index={price_change.index[-1]:now_time})

        # 第二步：确定分型：1是顶分型，2是底分型——用笔来确定合适的分型（但为了确保时序播放的性质，即不引入未来数据，对笔对应分型进行调整）
        # 满足条件1：出现两个或多个同性质的分型，连续顶分型，后面顶低于前面顶，则只保留前面；连续底分型，后面底低于前面，则只保留前面；
        # 否则均保留（即在后一个更高的顶分型出来前，无法对前一个顶分型进行剔除
        price_change['分型'] = np.nan
        price_change['顶点位置'] = np.nan
        change_date_list = sorted(price_change.index.to_list())

        index = 1
        while index < len(price_change) - 1:
            today = price_change.index[index]
            yesterday, tomorrow, Up_Type, Down_Type = get_entanglement_type(price_change, today) # 判断是顶分型还是底分型，
            # 如果既不是顶分型也是不底分型，则向前推进一日
            if (Up_Type == False) & (Down_Type == False):
                index +=1
                continue
            # 判断上一个是顶分型还是底分型
            last_type = price_change.loc[:today]['顶点位置'].dropna()
            # 1、如果是最开始，则保留该分型
            if len(last_type)>0:
                last_UpDownType = last_type.iloc[-1]
                last_date = last_type.index[-1]
                # 如果前一个是底分型 & 当前也是底分型
                if (last_UpDownType == -1) & (Down_Type == True):
                    # 如果前一个底比当前的底要低，则剔除该底部
                    if price_change.loc[last_date,'low'] < price_change.loc[today,'low']:
                        index += 1
                        continue
                # 如果前一个是顶分型 & 当前也是顶分型
                if (last_UpDownType == 1) & (Up_Type == True):
                    # 如果前一个顶比当前的顶要高，则剔除该顶部
                    if price_change.loc[last_date,'high'] > price_change.loc[today,'high']:
                        index += 1
                        continue
                # 如果前一个是顶分型，当前是底分型；或者前一个是底分型，当前是顶分型
                # 两个交易日之间必须间隔3日，否则不作数
                if ((last_UpDownType == 1) & (Down_Type == True)) | ((last_UpDownType == -1) & (Up_Type == True)):
                    if change_date_list.index(today) - change_date_list.index(last_date) <=3:
                        index += 1
                        continue

            if Up_Type == True:
                price_change.loc[[yesterday, today, tomorrow], '分型'] = 1,0,-1
                price_change.loc[today, '顶点位置'] = 1
            elif Down_Type == True:
                price_change.loc[[yesterday, today, tomorrow], '分型'] = -1,0,1
                price_change.loc[today, '顶点位置'] = -1
            index += 1

            # 按照缠论正规的历史周期来把控
            '''
            if len(last_type) > 0:
                last_UpDownType = last_type.iloc[-1]
                last_date = last_type.index[-1]
                # 如果前一个是顶分型 & 当前也是顶分型；当前顶分型比前一个顶分型顶点高，则删除前一个
                last_yesterday = list(price_change.index)[list(price_change.index).index(last_date) - 1]
                last_tomorrow = list(price_change.index)[list(price_change.index).index(last_date) + 1]

                if (last_UpDownType == 1) & (Up_Type == True):
                    if price_change.loc[last_date, 'high'] < price_change.loc[today, 'high']:
                        price_change.loc[last_date, '顶点位置'] = np.nan
                        price_change.loc[[last_yesterday, last_date, last_tomorrow], '分型'] = np.nan


                # 如果前一个是底分型 & 当前也是底分型；当前底分型比前一个底分型顶点低，则试产前一个
                if (last_UpDownType == -1) & (Down_Type == True):
                    if price_change.loc[last_date, 'low'] > price_change.loc[today, 'low']:
                        price_change.loc[last_date, '顶点位置'] = np.nan
                        price_change.loc[[last_yesterday, last_date, last_tomorrow], '分型'] = np.nan
            '''

        # 最后一步，返回原始数据结果
        FinallyResult = pd.concat([self.bench_high[bench_code], self.bench_low[bench_code],
                                   self.bench_open[bench_code], self.bench_close[bench_code]], axis=1)
        FinallyResult.columns = ['high', 'low', 'open', 'close']

        FinallyResult['分型'] =  price_change['分型']
        FinallyResult['顶点位置'] = price_change['顶点位置']

        FinallyResult['分型'] = FinallyResult['分型'].ffill()

        return FinallyResult

    # 1、北向资金成交额、成交额占比情况
    def north_amt_weight(self):
        north_money_weight = self.north_data['当日成交金额(人民币)'] * 1e8 / ((self.bench_amt['SZZZ'] + self.bench_amt['SZCZ']) * 1000)
        north_money_weight = north_money_weight.dropna().replace(0, np.nan).ffill()
        north_money = self.north_data['当日成交金额(人民币)'].replace(0, np.nan).ffill().rolling(10).mean()

        north_money_result = pd.concat([north_money_weight.rename('north_amt_weight'),north_money.rename('north_amt')],axis=1)
        return north_money_result


    # 1、北向资金的逆势交易行为
    def north_reverse_transaction(self,start_date,end_date,bench_code='CYBZ'):
        FinallyResult = self.Kline_Bench_Daily(bench_code)
        FinallyResult = FinallyResult[~FinallyResult['顶点位置'].ffill().isna()]
        bench_pct = self.bench_close.pct_change(1)
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0]  # 因为有很多交易日，北向资金不能交易，所以要剔除

        # 当前为上涨市场则为1，当前为下跌市场则为-1
        market_period = pd.DataFrame(index=FinallyResult.index,columns=['now_market','now_market_start','now_money_in',
                                                                        'before_market','before_market_start','before_market_end','before_money_in'])
        for date in FinallyResult.index:
            # 先获取当前市场的情况，和当前市场的开始点
            HaveDown_Market = FinallyResult.loc[:date, '顶点位置'].iloc[:-1].dropna()
            if len(HaveDown_Market) == 0:
                continue

            now_market, now_market_start = -HaveDown_Market.iloc[-1],HaveDown_Market.index[-1]

            if now_market == 1: # 表明这是向上的拐点
                begin_low,today_close = self.bench_low.loc[now_market_start,bench_code],  self.bench_close.loc[date,bench_code]
                if today_close < begin_low: # 如果当前的收盘价小于最低价，那么认为当前没有到拐点
                    HaveDown_Market = FinallyResult.loc[:HaveDown_Market[HaveDown_Market== 1].index[-1],'顶点位置'].dropna()
                    now_market, now_market_start = -HaveDown_Market.iloc[-1],HaveDown_Market.index[-1]
            elif now_market == -1: # 表明这是向下的拐点
                begin_high, today_close = self.bench_high.loc[now_market_start, bench_code], self.bench_close.loc[date, bench_code]
                if today_close > begin_high: # 如果当前的收盘价高于最高价，那么认为当前没有到拐点
                    HaveDown_Market = FinallyResult.loc[:HaveDown_Market[HaveDown_Market== -1].index[-1],'顶点位置'].dropna()
                    now_market, now_market_start = -HaveDown_Market.iloc[-1],HaveDown_Market.index[-1]
            # 先判断当前的最高价，和拐点的最高价

            market_period.loc[date,'now_market'] = now_market # 因为高点的顶点位置是1，所以当前是下跌，为-1；反之亦然
            #market_period.loc[date, 'now_market_start'] = now_market_start # 拐点为起始点

            # 如果拐点位置，指数是下跌的，拐点是向上的，那么拐点属于前一个，不属于今天
            if (bench_pct.loc[now_market_start, bench_code] < 0) & (market_period.loc[date,'now_market'] == 1):
                market_period.loc[date, 'now_market_start'] = self.date_list[self.date_list.index(now_market_start) + 1]  # 拐点为起始点
                market_period.loc[date, 'before_market_end'] = now_market_start
            elif  (bench_pct.loc[now_market_start, bench_code] > 0) & (market_period.loc[date,'now_market'] == 1):
                market_period.loc[date, 'before_market_end'] = self.date_list[self.date_list.index(now_market_start) - 1]  # 拐点为起始点
                market_period.loc[date, 'now_market_start'] = now_market_start

            if (bench_pct.loc[now_market_start, bench_code] < 0) & (market_period.loc[date,'now_market'] == -1):
                market_period.loc[date, 'before_market_end'] = self.date_list[self.date_list.index(now_market_start) - 1]  # 拐点为起始点
                market_period.loc[date, 'now_market_start'] = now_market_start
            elif (bench_pct.loc[now_market_start, bench_code] > 0) & (market_period.loc[date,'now_market'] == -1):
                market_period.loc[date, 'now_market_start'] = self.date_list[self.date_list.index(now_market_start) + 1]  # 拐点为起始点
                market_period.loc[date, 'before_market_end'] = now_market_start

            # 再获取前一个的市场情况，和前一个市场开始点
            before_position = HaveDown_Market[HaveDown_Market == now_market]
            if len(before_position) >0:
                before_market,before_market_start = -before_position.iloc[-1],before_position.index[-1]
                market_period.loc[date, 'before_market'] = before_market  # 这是前一个的位置
                market_period.loc[date, 'before_market_start'] =before_market_start

        market_period = market_period.iloc[1:]

        # 现在，根据市场划分的区间，进行北向资金的流入流出切割
        test_date_list = market_period['before_market_start'].dropna().index.to_list()

        money_result = pd.DataFrame(index=test_date_list,columns=['history_market','history_mkt_pct','history_north_in','history_corr',
                                                                  'now_market','now_mkt_pct','now_north_in','now_corr'])
        for date in test_date_list:
            if self.date_list.index(date) - self.date_list.index(market_period.loc[date,'now_market_start']) <=3:
                continue
            elif self.date_list.index(market_period.loc[date, 'before_market_end']) - self.date_list.index(market_period.loc[date, 'before_market_start']) <5:
                continue

            else:
                before_start = market_period.loc[date,'before_market_start']
                before_end = market_period.loc[date,'before_market_end']
                before_market_situation =  market_period.loc[date,'before_market']

                market_pct = self.bench_close.loc[before_end,bench_code] / self.bench_close.loc[get_pre_trade_date(before_start),bench_code] -1

                before_north_in = self.north_data.loc[get_pre_trade_date(before_start):before_end,'当日买入成交净额(人民币)']
                money_result.loc[date, 'history_market'] = before_market_situation
                money_result.loc[date, 'history_mkt_pct'] = market_pct
                money_result.loc[date,'history_north_in'] = before_north_in.sum()
                money_result.loc[date, 'history_corr'] = before_north_in.cumsum().corr(
                    self.bench_close.loc[get_pre_trade_date(before_start):before_end,bench_code])

                now_start = market_period.loc[date,'now_market_start']
                now_market_pct = self.bench_close.loc[date,bench_code] / self.bench_close.loc[get_pre_trade_date(now_start),bench_code] -1
                now_north_in = self.north_data.loc[get_pre_trade_date(now_start):date,'当日买入成交净额(人民币)']

                money_result.loc[date, 'now_market'] = market_period.loc[date,'now_market']
                money_result.loc[date, 'now_mkt_pct'] = now_market_pct
                money_result.loc[date, 'now_north_in'] = now_north_in.sum()
                money_result.loc[date, 'now_corr'] = now_north_in.cumsum().corr(
                    self.bench_close.loc[get_pre_trade_date(now_start):date,bench_code])


        a = pd.concat([market_period, money_result],axis=1)




    # 2、北向资金连续大幅净流入
    def north_always_in(self,start_date,end_date,days=20):
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0] # 因为有很多交易日，北向资金不能交易，所以要剔除
        # 计算北向资金大幅的净流入和净流出
        amt_average_in = new_north_data['当日买入成交净额(人民币)'].rolling(480).apply(lambda x: x[x > 0].dropna().median()).dropna()
        amt_average_out = new_north_data['当日买入成交净额(人民币)'].rolling(480).apply(lambda x: x[x < 0].dropna().median()).dropna()

        amt_std_in = new_north_data['当日买入成交净额(人民币)'].rolling(480).apply(lambda x: x[x > 0].dropna().std()).dropna()
        amt_std_out = new_north_data['当日买入成交净额(人民币)'].rolling(480).apply(lambda x: x[x < 0].dropna().std()).dropna()

        threshold_buy, threshold_sell = (amt_average_in + 1.5 * amt_std_in).dropna(), (amt_average_out - 1.5 * amt_std_out).dropna()

        x = new_north_data['当日买入成交净额(人民币)'].loc[:20210205].iloc[:-480][new_north_data['当日买入成交净额(人民币)'].loc[:20210205].iloc[:-480]>0]
        median = np.nanmedian(x)
        mad = np.nanmedian(abs(x - median))

        high = median + 3 * 1.4826 * mad
        low = median - 3 * 1.4826 * mad

        # 反转：当大幅净流入之后的净流出，和答复净流出之后的净流入，是买卖点的标志
        big_buy = (new_north_data['当日买入成交净额(人民币)'].loc[start_date:end_date] > threshold_buy.loc[start_date:end_date])
        big_sell = (new_north_data['当日买入成交净额(人民币)'].loc[start_date:end_date] < threshold_sell.loc[start_date:end_date])
        # （1）北向资金的纠错：如果在发生大幅净流出之后的5天内，发生了大幅净流入，且弥补了之前的流出，则认为行情没有结束
        buy_point = big_buy[(big_sell.rolling(5).sum() >= 1) & (big_buy)]
        sell_point = big_sell[(big_buy.rolling(5).sum() >= 1) & (big_sell)]

        index_pct = self.index_close.pct_change(10).shift(-10)
        index_pct.loc[big_buy[(big_sell.rolling(10).sum() >= 1) & (big_buy)].index]





        big_buy[big_buy==True]
        big_sell[big_sell==True]













        a = (continue_money >=5).shift(1).loc[start_date:end_date] & (new_north_data['当日买入成交净额(人民币)'].loc[start_date:end_date]
                                                                      < threshold_sell.loc[start_date:end_date])
        a[a==True]

        period_money_date = (new_north_data['当日买入成交净额(人民币)'] > 0).rolling(days).sum() / days










result = pd.concat([self.index_close[['SZZZ','CYBZ','wind_A','avg_A']],continue_money.rename('continue_money'),
                   period_money_date.rename('period_money_date')],axis=1).sort_index()
result.to_pickle('E:/north_in.pkl')


################################################################ 北向资金数据 #####################################################
start_date,end_date = 20160101,20220916
self = North_Money_Data(start_date,end_date)

north_money_result = self.north_amt_weight()
#north_money_result.to_pickle('E:/north_in.pkl')



result = pd.concat([index_close[['SZZZ','CYBZ','wind_A']],north_money.rename('north_amt')],axis=1).sort_index()




######################################################### 计算北向资金的各种指标 ###################################################
# 1、北向资金连续净流入天数




# 2、北向资金最近N日的净流入天数
days = 20
average_cashflow = north_buy_weight.rolling(480).mean().dropna()

sum_in = (new_north_data['当日买入成交净额(人民币)'] > new_north_data['当日买入成交金额(人民币)'] * 0.05).rolling(days).sum() / days
sum_out = (new_north_data['当日买入成交净额(人民币)'] < new_north_data['当日卖出成交金额(人民币)'] * 0.05).rolling(days).sum() / days

# 3、北向资金的大幅净流入

big_buy_count = (new_north_data.loc[threshold_buy.index, '当日买入成交净额(人民币)'] >= threshold_buy).rolling(days).sum().dropna() / days
big_sell_count = (new_north_data.loc[threshold_buy.index, '当日买入成交净额(人民币)'] <= threshold_sell).rolling(days).sum().dropna() / days
net_buy = big_buy_count - big_sell_count # 净单日大幅流入

# 4、给定周期中，流入金额和流出金额的比值
cash_flow_ratio =  - new_north_data['当日买入成交净额(人民币)'].rolling(20).apply(lambda x:x[x>0].dropna().mean()) / new_north_data['当日买入成交净额(人民币)'].rolling(20).apply(lambda x:x[x<0].dropna().mean())




###################### 即明确什么时候关心北向资金的特征 ###########################################
def get_result_df(trade_date,bench='SZZZ',future_day = 20):
    test_result = pd.DataFrame(index=trade_date,columns=['position','before_profit','before_loss','max_up','max_down','max_up_time','max_down_time'])
    # 历史低点到触发点的涨幅
    close_bench = index_close[bench]
    for date in trade_date:
        if date in close_bench.index:
            position = close_bench.loc[:date].iloc[-240:].rank(pct=True).loc[date]
            before_profit = close_bench.loc[date] / close_bench.loc[:date].iloc[-21:].min() -1
            before_loss = close_bench.loc[date] / close_bench.loc[:date].iloc[-21:].max() - 1
            max_up = close_bench.rolling(future_day).max().shift(-future_day).loc[date] / close_bench.loc[date] - 1
            max_down = close_bench.rolling(future_day).min().shift(-20).loc[date] / close_bench.loc[date] - 1
            max_up_time = close_bench.loc[date:].iloc[:20].argmax()+1
            max_down_time = close_bench.loc[date:].iloc[:20].argmin() + 1

            test_result.loc[date] = position, before_profit, before_loss, max_up, max_down, max_up_time, max_down_time
    return test_result.dropna(how='all')

# 1、当指数下跌，而北向资金逆势加仓时，是否具备较好的拐点机会。
prod_money = north_data['累计买入成交净额(人民币)'].replace(0,np.nan).ffill()
corr = rolling_corr(pd.DataFrame(prod_money.loc[start_date:end_date]),index_close[['SZZZ']].loc[start_date:end_date],window=20).dropna()
north_money_in = north_data['当日买入成交净额(人民币)'].rolling(20).sum()
north_money_rank = ts_rank(north_money_in[],rol_day=60)

buy_position = pd.Series(index = corr.index)
trade_start = 0
for date in corr.index:
    # 判断背离1：负相关性较高，如果资金净流入则为市场下跌净流入；如果负相关性较低则为市场上涨资金净流出
    if corr.loc[date,'累计买入成交净额(人民币)'] <= -0.6:
        if (north_money_in.loc[date] > 20):
            money_in_weight = north_money_in.loc[:date].iloc[-60:][north_money_in.loc[:date].iloc[-60:] > 0].rank(pct=True).loc[date]
            if (money_in_weight > 0.6):
                trade_start = 1
        elif (north_money_in.loc[date] < -20):
            money_out_weight = abs(north_money_in.loc[:date][north_money_in.loc[:date] < 0].iloc[-60:]).rank(pct=True).loc[date]
            if (money_in_weight > 0.6):
                trade_start = -1
    # 判断背离2：可以不需要负相关性特别高，只需要相关性下降幅度特别大即可
    if ((corr.loc[:date,'累计买入成交净额(人民币)'].iloc[-40:].max() - corr.loc[date,'累计买入成交净额(人民币)']) > 0.5) & \
            (corr.loc[date,'累计买入成交净额(人民币)'] < 0.2):
        period_money_in = north_data['当日买入成交净额(人民币)'].loc[corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-40:].idxmax():date]
        period_money_in = period_money_in[period_money_in != 0]
        if (north_money_in.loc[date] > 20): # 如果相关性下降期间，资金是在持续净流入的
            money_in_weight = north_money_in.loc[:date].iloc[-60:][north_money_in.loc[:date].iloc[-60:] > 0].rank(pct=True).loc[date]
            # 如果资金流持续净流入，且净流入的幅度较大
            if ((period_money_in.diff() >0).sum() / len(period_money_in) > 0.5) & (money_in_weight > 0.7):
                trade_start = 1
        elif (north_money_in.loc[date] < -20): # 如果相关性下降，资金持续净流出
            money_out_weight = \
            abs(north_money_in.loc[:date][north_money_in.loc[:date] < 0].iloc[-60:]).rank(pct=True).loc[date]
            if ((period_money_in.diff() <0).sum() / len(period_money_in) > 0.5) & (money_out_weight > 0.7):
                trade_start = -1

    # 当出现背离现象，再转为一致时：
    if (corr.loc[date,'累计买入成交净额(人民币)'] >= 0.6) & (trade_start != 0):
        # 如果北向资金和指数同时上涨，
        if (trade_start == 1) & (north_money_in.loc[date] > 0):
            buy_position.loc[date] = 1
            trade_start = 0
        if (trade_start == -1):
            buy_position.loc[date] = -1
            trade_start = 0

    elif  ((corr.loc[date,'累计买入成交净额(人民币)'] - corr.loc[:date,'累计买入成交净额(人民币)'].iloc[-40:].min()) > 0.5):
        # 如果相关系数迅速回暖，如果之前是1则是买点，之前是-1则是卖点
        if (trade_start == 1) & (north_money_in.loc[date] > 0):
            buy_position.loc[date] = 1
            trade_start = 0
        if (trade_start == -1):
            buy_position.loc[date] = -1
            trade_start = 0

buy_position = buy_position.dropna()



# 判断买点1：当前负相关性较高，但是资金是净流入的，表明北向资金在抄底
buy_position1 = pd.Series(index = corr.index)
watch_time,watch_date = 0, np.nan
for date in corr.index:
    if (corr.loc[date,'累计买入成交净额(人民币)'] <= -0.6) & (north_money_in.loc[date] > 0):
        # 先寻找到相关系数最高点，从最高点到今天的区间，累计净流入情况，和净流入天数
        period_north = north_data['当日买入成交净额(人民币)'].loc[corr.loc[:date,'累计买入成交净额(人民币)'].iloc[-20:].idxmax():date]
        period_north = period_north[period_north !=0 ]
        period_north_useful = period_north[abs(period_north) >= 5]
        money_in_weight = (period_north_useful > 0 ).sum() / len(period_north_useful)
        period_north_in = period_north.sum()

        if (money_in_weight > 0.6) & (period_north_in > 20):
            watch_time = 1
            buy_position1.loc[date] = 0
            watch_date = date

    # 北向资金在市场下跌时进行抄底，那么当北向资金和市场相关性回升，即北向资金和市场同时向上时，则是买入时机：
    corr_turn = corr.loc[date,'累计买入成交净额(人民币)'] - corr.loc[:date,'累计买入成交净额(人民币)'].iloc[-30:].min()

    if (watch_time == 1) & ((corr_turn > 0.5) | (corr.loc[date,'累计买入成交净额(人民币)'] > 0.6)) & (corr.loc[date,'累计买入成交净额(人民币)'] > 0 ):
        # 如果这个时候资金同样是净流入的，那么就是买点
        period_north = north_data['当日买入成交净额(人民币)'].loc[corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-20:].idxmin():date]
        period_north = period_north[period_north != 0]
        money_in_weight = (period_north > 0).sum() / len(period_north)
        period_north_in = period_north.sum()

        # 必须是前几天达成的，不能是从高位i回落
        first_get = corr.loc[corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-20:].idxmax():date].rank(pct=True).loc[date]['累计买入成交净额(人民币)']>0.7

        if (money_in_weight > 0.6) & (period_north_in > 50) & first_get:
            if len(corr.loc[watch_date:date]) <= 60:
                watch_time = 0
                watch_date = np.nan
                buy_position1.loc[date] = 1
            else:
                watch_time = 0
                watch_date = np.nan


# 判断买点2：不需要负相关性特别高，只需要相关性大幅下降，同时北向资金在抄底
buy_position2 = pd.Series(index = corr.index)
watch_time,watch_date = 0, np.nan
for date in corr.index:
    if ((corr.loc[:date,'累计买入成交净额(人民币)'].iloc[-40:].max() - corr.loc[date,'累计买入成交净额(人民币)']) > 0.5) & \
            (corr.loc[date,'累计买入成交净额(人民币)'] < 0.2):
        # 条件1：从高位回落，不能是从高位回落，又从低位起来，因此要找到最近的一个高位
        drop_high_down = corr.loc[corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-40:].idxmax():date, '累计买入成交净额(人民币)'] - \
                         corr.loc[date, '累计买入成交净额(人民币)']

        high_position = drop_high_down[drop_high_down > 0.5].diff()
        if (high_position>0).sum() == 0:
            down_start_date = drop_high_down.index[0]
        else:
            down_start_date = high_position[high_position>0].index[-1]
        if (corr.loc[date, '累计买入成交净额(人民币)'] - corr.loc[down_start_date:date, '累计买入成交净额(人民币)'].min()) < 0.3:
            # 第二点，需要判断一下，在下降周期，资金是不是净流入的
            corr.loc[down_start_date : date]
            period_north = north_data.loc[down_start_date : date, '当日买入成交净额(人民币)'].iloc[1:]
            period_north_useful = period_north.loc[abs(period_north) >= 5]
            money_in_weight = (period_north_useful > 0).sum() / len(period_north_useful)
            period_north_in = period_north.sum()

            if ((money_in_weight > 0.6) & (period_north_in > 20)) or ((money_in_weight >= 0.5) & (period_north_in > 100)):
                watch_time = 1
                buy_position2.loc[date] = 0
                watch_date = date

    # 北向资金在市场下跌时进行抄底，那么当北向资金和市场相关性回升，即北向资金和市场同时向上时，则是买入时机：
    corr_turn = corr.loc[date, '累计买入成交净额(人民币)'] - corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-30:].min()

    if (watch_time == 1) & ((corr_turn > 0.5) | (corr.loc[date, '累计买入成交净额(人民币)'] > 0.6)) & (
            corr.loc[date, '累计买入成交净额(人民币)'] > -0.1):
        # 如果这个时候资金同样是净流入的，那么就是买点
        period_north = north_data['当日买入成交净额(人民币)'].loc[corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-20:].idxmin():date]
        period_north = period_north[period_north != 0]
        money_in_weight = (period_north > 0).sum() / len(period_north)
        period_north_in = period_north.sum()

        # 必须是前几天达成的，不能是从高位i回落
        first_get = corr.loc[max(corr.loc[:date, '累计买入成交净额(人民币)'].iloc[-20:].idxmax(),watch_date):date].rank(pct=True).loc[date][
                        '累计买入成交净额(人民币)'] > 0.7

        if (money_in_weight > 0.6) & (period_north_in > 50) & first_get:
            if len(corr.loc[watch_date:date]) <= 60:
                watch_time = 0
                watch_date = np.nan
                buy_position2.loc[date] = 1
            else:
                watch_time = 0
                watch_date = np.nan

buy_position2 = buy_position2.dropna()
buy_position2[buy_position2 == 1]


buy_position.dropna().loc[:20200221]
date = 20150522
核心：相比于前一段时间也得是净流入，不能前一段时间流入多，这段时间流入少
# 北向资金走势和市场走势无关，并且北向资金从净流入转为净流出，和从净流出转为净流入，并且是大幅转向
corr = corr
a =pd.concat([corr.rename(columns = {'累计买入成交净额(人民币)':'corr'}),north_data['当日买入成交净额(人民币)'].rolling(20).sum()],axis=1).dropna().sort_index()
b = ((a['corr']>-0.2) & (a['corr']<0.2)).rolling(3).sum()==3
c = (a['当日买入成交净额(人民币)'].rolling(20).max() - a['当日买入成交净额(人民币)'] > 50) & (a['当日买入成交净额(人民币)']<20)


a[b & c]


(a['当日买入成交净额(人民币)'].rolling(20).max() - a['当日买入成交净额(人民币)'] < 50).loc[20220516]

result = pd.concat([corr['累计买入成交净额(人民币)'].rename('corr'),index_close['CYBZ'],index_close['SZZZ'],index_close['SZ50']],axis=1)
result.to_pickle('E:/north_in.pkl')



north_data['累计买入成交净额(人民币)'].loc[:20160124].iloc[-20:].corr(index_close['SZZZ'].loc[:20160124].iloc[-20:])

north_data.loc[20151225]


# 1、当北向资金出现连续的净流入，净流出时
max_continue_in = continue_in >= 5
max_continue_out = continue_out >= 5

max_continue = max_continue_in.copy().astype(int)
max_continue[max_continue_out >0] = -1


# 买点1：第一次出现信号时为买入点
trade_date = (max_continue_in == True) & (max_continue_in.shift(1) == False)
trade_date = trade_date[trade_date==True].index.to_list()


a = get_result_df(trade_date,bench='CYBZ',future_day = 20)
a['max_up'].median()
a['max_up_time'].median()
a['max_down'].median()
a['max_down_time'].median()
a = a.sort_values(by='before_profit').astype(float).round(4)


a[a['before_profit'] >=0.05]


result = pd.concat([max_continue.rename('conintue_inout'),index_close['CYBZ'],index_close['SZZZ'],index_close['SZ50']],axis=1)
result.to_pickle('E:/north_in.pkl')



# 1、当市场指数在进行周期性上涨时，北向资金的大幅流入是行情延续的标志（即新进入资金延续了行情）
# 北向资金的大幅流入：①北向资金的流入天数＞50%，②大幅净流入＞大幅净流出 ③ 北向资金净流入水平高
north_fund_net_in_rank = ts_rank(north_fund_net_in,rol_day=480)

north_in_date = (always_in >0.5) & (net_buy >=0) & (north_fund_net_in_rank >0.6) & (north_fund_net_in>0)








north_fund_net_in_rank.loc[20210501:20210615]
north_fund_net_in.loc[20220701:20220916]





test_result =


test_result.sort_values(by='before_profit')






















