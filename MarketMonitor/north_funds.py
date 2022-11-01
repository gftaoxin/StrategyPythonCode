import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import requests,json,datetime,time,sys
from dataApi import getData,tradeDate,stockList
from BasicData.local_path import *
from usefulTools import *

# 长下影线
def long_below_line(bench_open, bench_close, bench_high, bench_low,low_weight=0.4,up_weight=0.3):
    low_line_long = pd.concat([bench_open, bench_close]).min(level=0) - bench_low
    up_line_long = bench_high - pd.concat([bench_open, bench_close]).max(level=0)
    long_low_line = (low_line_long / (bench_high - bench_low) > low_weight) & (up_line_long / (bench_high - bench_low) < up_weight)
    long_low_line = long_low_line[['SZZZ', 'CYB', 'CYBZ', 'wind_A', 'avg_A']]

    return long_low_line

def long_high_line(bench_open, bench_close, bench_high, bench_low):
    low_line_long = pd.concat([bench_open, bench_close]).min(level=0) - bench_low
    up_line_long = bench_high - pd.concat([bench_open, bench_close]).max(level=0)
    long_up_line = (low_line_long / (bench_high - bench_low) < 0.3) & (up_line_long / (bench_high - bench_low) > 0.4)
    long_up_line = long_up_line[['SZZZ', 'CYB', 'CYBZ', 'wind_A', 'avg_A']]

    return long_up_line

# 函数1：获取某个日期的前后两日
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
    # 1、数据说明：北向资金成交额、成交额占比情况
    def north_amt_weight(self):
        north_money_weight = self.north_data['当日成交金额(人民币)'] * 1e8 / ((self.bench_amt['SZZZ'] + self.bench_amt['SZCZ']) * 1000)
        north_money_weight = north_money_weight.dropna().replace(0, np.nan).dropna()
        north_money = self.north_data['当日成交金额(人民币)'].replace(0, np.nan).dropna().rolling(10).mean()

        north_money_result = pd.concat([north_money_weight.rename('north_amt_weight'),north_money.rename('north_amt')],axis=1)
        return north_money_result
    # 2、数据计算：北向资金大幅度净流入
    def north_big_in(self,start_date,end_date):
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        # 计算北向资金大幅的净流入和净流出
        north_big_in = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] >0].rolling(252,min_periods=1).quantile(0.9)
        north_big_out = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] <0].rolling(252,min_periods=1).quantile(0.1)
        north_big_in = north_big_in.apply(lambda x: 20 if x < 20 else x).reindex(new_north_data.index).ffill()
        north_big_out = north_big_out.apply(lambda x: -20 if x > -20 else x).reindex(new_north_data.index).ffill()

        north_middle_in = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] > 0].rolling(252,min_periods=1).quantile(0.8)
        north_middle_out = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] < 0].rolling(252,min_periods=1).quantile(0.2)
        north_middle_in = north_middle_in.apply(lambda x: 20 if x < 20 else x).reindex(new_north_data.index).ffill()
        north_middle_out = north_middle_out.apply(lambda x: -20 if x > -20 else x).reindex(new_north_data.index).ffill()

        north_limit = pd.concat([north_big_in.rename('north_big_in'),north_big_out.rename('north_big_out'),north_middle_in.rename('north_middle_in'),north_middle_out.rename('north_middle_out')],axis=1)

        return north_limit.loc[start_date:end_date]
    # 1、事件情况1：北向资金的连续净流入
    def north_always_in(self,flow_in_days,start_date,end_date,bench = 'CYBZ'):
        # 基础数据采集
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0].loc[get_pre_trade_date(start_date,252):end_date]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        north_limit = self.north_big_in(get_pre_trade_date(start_date,252),end_date)
        # 获取站在当天看，净卖出多少算是结束点:①长期平均净卖出，②短期平均净卖出，③日均平均成交额
        north_start_out = pd.concat([abs(new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] < 0].rolling(252, min_periods=1).mean()),
                   abs(new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] < 0].rolling(60, min_periods=1).mean()),
                   abs(new_north_data['当日买入成交净额(人民币)'].rolling(20, min_periods=1).mean()).loc[new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] < 0].index]]).max(level=0)
        north_start_out = north_start_out.apply(lambda x:min(max(x,20),40))
        north_start_out = north_start_out.reindex(self.date_list).ffill().loc[get_pre_trade_date(start_date,252):end_date]
        # 站在当天看，过去一年内平局净流入的金额，和中位数净流入金额
        north_middle_in = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] > 0].rolling(252,min_periods=1).mean()
        north_middle_in = north_middle_in.reindex(new_north_data.index).ffill()
        north_midian_in = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] > 0].rolling(252, min_periods=1).median().apply(lambda x:min(max(x,20),40))
        north_midian_in = north_midian_in.reindex(new_north_data.index).ffill()

        # 计算连续净流入的天数: ①在进入连续净流入必须是5日内的连续净流入，后续在考虑净流入是否结束时，如果下一日净流出＜20亿，则仍然算作净流入
        continue_in = pd.DataFrame(index=new_north_data.index,columns=['continuous_in_days','count_days','big_in','middle_in','flow_middle_in_raito','flow_in_ratio','flow_in_money'])
        i,j = 0,0
        for date in continue_in.index:
            # 先定义净流入情况
            if (new_north_data.loc[date, '当日买入成交净额(人民币)'] > 0) & ((i/j if j>0 else 1)>0.7):  # 如果当日是净流入的，那么天数就+1
                i += 1
                j += 1
            else:
                # 如果当日不是净流入的，但如果已经累计净流入超过5日，则看净流出水平，如果水平比较低，那么不算净流出，但不增加净流入天数
                if (i >=flow_in_days) & ((i/j if j>0 else 1)>0.7) & (new_north_data.loc[date, '当日买入成交净额(人民币)'] > -north_start_out.loc[date]):
                    j += 1
                else:
                    i,j = 0, 0
            # 再定义净流出情况
            if j !=0:
                north_inflow_money = new_north_data.loc[:date, '当日买入成交净额(人民币)'].iloc[-j:]
                big_in = (north_inflow_money > north_limit.loc[north_inflow_money.index, 'north_big_in']).sum()
                middle_in = (north_inflow_money > north_middle_in.loc[north_inflow_money.index]).sum()
                netflow_in_ratio = (north_inflow_money > north_midian_in.reindex(north_inflow_money.index)).sum() / j
                flow_in_ratio = i / j

                continue_in.loc[date] = i, j, big_in, middle_in, netflow_in_ratio, flow_in_ratio, north_inflow_money.mean()

            else:
                continue_in.loc[date] = i, j, 0, 0, 0, 0, 0


                # 在净流入之外，要满足两个条件

        # 看好情况1：净流入超过5天，且其中大幅净流入天厨超过1天，或者中幅净流入天数超过3天，且净买入天数较多，
        position1 = (continue_in['continuous_in_days'] >= flow_in_days) & \
        ((continue_in['big_in'] >= continue_in['continuous_in_days'].apply(lambda x: math.ceil(x/10))) | \
        (continue_in['middle_in'] >= continue_in['continuous_in_days'].apply(lambda x: math.ceil(x / 10+3)))) & \
        ((continue_in['flow_middle_in_raito'] >= 0.5) | (continue_in['flow_in_ratio'] >= 0.8) | (continue_in['flow_in_money'] > north_middle_in))
        # 看好情况2：净流入超过10日，且其中大幅净流入天厨超过1天，或者中幅净流入天数超过3天，或者当日大幅净买入即可
        position2 = (continue_in['continuous_in_days'] >= 10) & \
                    ((continue_in['flow_in_money'] > north_middle_in) | ((continue_in['big_in'] >= continue_in['continuous_in_days'].apply(lambda x: math.ceil(x/10))) | \
        (continue_in['middle_in'] >= continue_in['continuous_in_days'].apply(lambda x: math.ceil(x / 10+3)))))
        # 看好情况3：净流入虽然低于5天，但是大幅净流入比例较高
        min_continous = continue_in['continuous_in_days'].apply(lambda x:max(math.ceil(flow_in_days/2),x))
        position3 = (continue_in['continuous_in_days'] >= min_continous) & \
                    ((continue_in['big_in'] >= min_continous/2) | (continue_in['middle_in'] >= min_continous * 0.8))

        continue_in['continuous_in_signal'] = position1 | position2 | position3
        continue_in = continue_in.reindex(self.date_list).ffill().loc[start_date:end_date]

        # 开始划定，参与交易的行为
        norht_in_signal = pd.DataFrame(index= continue_in.index,columns=['buy_signal','market_position'])
        date = continue_in.index[0]
        while date <= continue_in.index[-1]:
            # 先判断市场环境：从高点下跌的幅度，和从低点反弹的幅度，比较接近；那么认为市场已经反弹企稳，当前并非左侧
            north_in_period = self.bench_close.loc[:date,['SZZZ','CYBZ','CYB','wind_A','avg_A']].iloc[-int(continue_in.loc[date,'continuous_in_days'])-1:]
            max_down = north_in_period.loc[date] / north_in_period.max() - 1
            max_up = north_in_period.loc[date] / north_in_period.min() - 1
            north_type = 'left' if ((max_down < - 0.03) & (max_up  < abs(max_down/2))).sum() >= 2 else 'right' # 如果当前市场的最大涨幅和最大跌幅都 ＜ 5%，则认为是震荡区间

            if (north_type == 'right') & (continue_in.loc[date, 'continuous_in_signal'] == True):
                # 如果当前是右侧时间节点：那么北向资金的连续净流入期间，均为参与市场的时间点，即市场没有较高的风险
                buy_date = date
                sell_date = get_pre_trade_date(buy_date, -1)
                norht_in_signal.loc[buy_date] = 1,north_type
                while continue_in.loc[sell_date, 'continuous_in_signal'] == True:
                    norht_in_signal.loc[sell_date] = 1,north_type
                    sell_date = get_pre_trade_date(sell_date, -1)
                date = get_pre_trade_date(sell_date, -1)
            elif (north_type == 'left') & (continue_in.loc[date, 'continuous_in_signal'] == True) :
                # 如果当前时间点为左侧，那么表明北向资金当前是在抄底，不急于和北向资金一起抄底；只有当北向资金不再净流入时，才开始抄底
                buy_date = None
                for cheat_date in new_north_data.loc[date:][:5].index:
                    if (new_north_data.loc[cheat_date,'当日买入成交净额(人民币)'] < north_limit.loc[cheat_date,'north_middle_out']):
                        buy_date = cheat_date
                        sell_date = get_pre_trade_date(buy_date, -10)
                        norht_in_signal.loc[buy_date:sell_date].iloc[:-1] = 1,north_type
                        while continue_in.loc[sell_date, 'continuous_in_signal'] == True:  # 如果在10天北水继续净流入，那么就一直持续到北水净流入结束
                            norht_in_signal.loc[sell_date] = 1,north_type
                            sell_date = get_pre_trade_date(sell_date, -1)
                        date = get_pre_trade_date(sell_date, -1)
                        break
                if buy_date == None:
                    date = get_pre_trade_date(date, -1)

            else:
                date = get_pre_trade_date(date, -1)

        # 开始拆分，北向资金的连续净流入周期的行为：
        market_in_position = pd.DataFrame(columns=['buy_date', 'sell_date', 'period_days', 'bench_pct', 'maxdown', 'north_type'])
        date, i = continue_in.index[0],0
        while date <= continue_in.index[-1]:
            if norht_in_signal.loc[date,'buy_signal'] == 1:
                north_type = norht_in_signal.loc[date,'market_position']
                buy_date = date
                sell_date = min(get_pre_trade_date(buy_date, -1), end_date)
                while norht_in_signal.loc[sell_date,'buy_signal'] == 1:
                    if sell_date == end_date:
                        break
                    sell_date = get_pre_trade_date(sell_date, -1)

                priod_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[buy_date, bench] - 1
                maxdown = self.bench_close.loc[buy_date:sell_date, bench].min() / self.bench_close.loc[buy_date, bench] - 1
                market_in_position.loc[i] = buy_date, sell_date, get_trade_date_interval(sell_date,buy_date), priod_pct, maxdown, north_type
                i +=1
                date = get_pre_trade_date(sell_date, -1)
            else:
                date = get_pre_trade_date(date,-1)

        norht_in_signal['buy_signal'] = norht_in_signal['buy_signal'].fillna(0)

        norht_in_signal = pd.concat([continue_in[['continuous_in_days','count_days','big_in','middle_in']],norht_in_signal],axis=1)

        return norht_in_signal, market_in_position
    # 2、事件情况2：北向资金的连续净流出（摆脱指数的影响）
    def north_always_out(self,flow_out_days,start_date,end_date,bench = 'CYBZ'):
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0].loc[get_pre_trade_date(start_date, 252):end_date]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        north_limit = self.north_big_in(get_pre_trade_date(start_date, 252), end_date)
        north_big_flow = ((new_north_data['当日买入成交净额(人民币)'] > north_limit['north_middle_in']) * 1 + (new_north_data['当日买入成交净额(人民币)'] < north_limit['north_big_out']) * -1).loc[start_date:end_date]
        north_big_flow = north_big_flow.reindex(self.date_list).loc[start_date:end_date].fillna(0)
        # 站在当天看，过去一年内平局净流出的金额，和中位数净流出金额
        north_middle_out = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] < 0].rolling(252,min_periods=1).mean().apply(lambda x:min(x,-20))
        north_middle_out = north_middle_out.reindex(new_north_data.index).ffill()
        # 计算连续净流出的天数：①在进行连续净流出必须是5日内的连续净流出，后续在考虑净流出是否结束时，如果连续3日净流出较低，即认为结束。
        continue_out, i, j = pd.DataFrame(index=new_north_data.index,columns=['continuous_out_days', 'count_days','big_out','middle_out','flow_out_money']), 0, 0
        for date in continue_out.index:
            if (new_north_data.loc[date, '当日买入成交净额(人民币)'] < 0) & (i == j):  # 如果当日是净流出的，那么天数就+1
                i += 1
                j += 1
            else:
                i, j = 0, 0
            if j !=0:
                north_inflow_money = new_north_data.loc[:date, '当日买入成交净额(人民币)'].iloc[-j:]
                big_out = (north_inflow_money < north_limit.loc[north_inflow_money.index, 'north_big_out']).sum()
                middle_out = (north_inflow_money < north_middle_out.loc[north_inflow_money.index]).sum()
                continue_out.loc[date] = i, j, big_out, middle_out, north_inflow_money.mean()
            else:
                continue_out.loc[date] = i, j, 0, 0, 0
        # 指数是否有长下影线
        long_low_line = long_below_line(self.bench_open, self.bench_close, self.bench_high, self.bench_low)
        bench_pct = (self.bench_close.pct_change()[['SZZZ', 'CYB', 'CYBZ', 'wind_A', 'avg_A']] > 0.015) # 指数大涨
        # 当日是否大幅净流入 & 大涨：只有大涨的大幅净流入才算数
        net_in_up = (north_big_flow == 1) & ((bench_pct | long_low_line).sum(axis=1) >=3)
        # 看空情况1：净流出超过5天
        position1 = (continue_out['continuous_out_days'] >=flow_out_days)
        # 看空情况2：净流出虽然低于5天，但是大幅净流出比例较高
        min_continous = continue_out['continuous_out_days'].apply(lambda x: max(math.ceil(flow_out_days / 2), x))
        position2 = (continue_out['continuous_out_days'] >= min_continous) & ((continue_out['big_out'] >= min_continous / 2) | (continue_out['middle_out'] >= min_continous * 0.8))

        continue_out['continuous_out_signal'] = position1 | position2
        continue_out = continue_out.reindex(self.date_list).ffill().loc[start_date:end_date]

        # 开始制定参与计划
        north_median_out = new_north_data['当日买入成交净额(人民币)'][new_north_data['当日买入成交净额(人民币)'] < 0].rolling(252,min_periods=1).median().apply(lambda x: min(x, -20))
        north_median_out = north_median_out.reindex(new_north_data.index).ffill()

        norht_out_signal = pd.DataFrame(index=continue_out.index, columns=['sell_signal','short_signal'])
        date = continue_out.index[0]
        while date <= continue_out.index[-1]:
            # 无需判断市场环境，但是要关注市场是否存在下影线
            if (continue_out.loc[date, 'continuous_out_signal'] == True):
                # 判断当前是否存在下影线，如果有下影线；则推迟一天
                if long_low_line.loc[date].sum() >= 3:
                    date = get_pre_trade_date(date,-1)
                    continue
                else:
                    # 如果当日没有下影线，则开始看空，看空到①出现了长下影线 或②连续3日没有大幅净流出 或者③当日没有大幅度净流入 或者④从日内-2%以下最低点上涨超过2%
                    norht_out_signal.loc[date] = -1,-1
                    sell_date = min(get_pre_trade_date(date,-1),end_date)
                    no_out_days = (new_north_data.loc[:sell_date,'当日买入成交净额(人民币)'].iloc[-3:] >= north_median_out.loc[new_north_data.loc[:sell_date,'当日买入成交净额(人民币)'].iloc[-3:].index]).sum()
                    while (long_low_line.loc[sell_date].sum() < 3) & (no_out_days <3) & (net_in_up.loc[sell_date] == False):
                        # 如果当前北向资金没有参与市场，则为np.nan
                        norht_out_signal.loc[sell_date,'sell_signal'] = -1
                        if sell_date == end_date:
                            break
                        sell_date = min(get_pre_trade_date(sell_date, -1), end_date)
                    # 此时，后面连续的True都不顶用
                    while (continue_out.loc[sell_date, 'continuous_out_signal'] == True):
                        if sell_date == end_date:
                            break
                        sell_date = min(get_pre_trade_date(sell_date, -1), end_date)
                    date = sell_date
                    if date ==  end_date:
                        break
            else:
                date = get_pre_trade_date(date, -1)

        # 开始计算看空周期的结果
        market_out_position = pd.DataFrame(columns=['buy_date', 'sell_date', 'period_days', 'bench_pct', 'maxup','maxdown'])
        date, i = continue_out.index[0], 0
        while date <= continue_out.index[-1]:
            if (norht_out_signal.loc[date, 'sell_signal'] == -1):
                buy_date = date
                sell_date = min(get_pre_trade_date(buy_date, -1),end_date)
                while norht_out_signal.loc[sell_date, 'sell_signal'] == -1:
                    if sell_date == end_date:
                        break
                    sell_date = get_pre_trade_date(sell_date, -1)

                priod_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[buy_date, bench] - 1
                maxup = self.bench_close.loc[buy_date:sell_date, bench].max() / self.bench_close.loc[buy_date, bench] - 1
                maxdown = self.bench_close.loc[buy_date:sell_date, bench].min() / self.bench_close.loc[buy_date, bench] - 1

                market_out_position.loc[i] = buy_date, sell_date, get_trade_date_interval(sell_date,buy_date), priod_pct, maxup, maxdown
                i += 1
                date = get_pre_trade_date(sell_date, -1)
            else:
                date = get_pre_trade_date(date, -1)

        # 北向资金净流出时，如果市场并没有跟随下跌，那么当北向资金净流出结束后，开始大幅净流入
        # 满足：大幅净流入 & 市场上涨强度不够，如果大幅净流入市场大涨，则等到中等净流出时再卖
        # 停止看空：北向资金5日内有3日不再净流出，且期间没有大幅净流出；或者北向资金单日大幅净流入 & 市场大涨
        norht_out_signal['turn_short_position'] = 0
        for date in norht_out_signal['sell_signal'][norht_out_signal['sell_signal'] == -1].index:
            if (norht_out_signal.loc[date, 'sell_signal'] == -1) & (continue_out.loc[date, 'continuous_out_days'] >= 3):
                north_in_period = self.bench_close.loc[:date, ['SZZZ', 'CYBZ', 'CYB', 'wind_A', 'avg_A']].iloc[
                                  -int(continue_out.loc[date, 'continuous_out_days']) - 1:]
                max_down = north_in_period.loc[date] / north_in_period.max() - 1
                max_up = north_in_period.loc[date] / north_in_period.min() - 1
                # 当跌幅超过-3%，且反弹低于回撤一半的数量＜2时，认为当前市场为右侧（即尚未下跌）
                if ((max_down < - 0.03) & (max_up < abs(max_down / 2))).sum() <= 2:
                    buy_date = None
                    # 此时，如果在北向资金停止净流出之后的5日内，北向资金出现了大幅度净流入
                    not_out_days = (continue_out.loc[date:, 'continuous_out_signal'] == True) | (norht_out_signal.loc[date:, 'sell_signal'] == -1)
                    if (not_out_days == False).sum() == 0:
                        continue
                    not_out_days = not_out_days[not_out_days == False].index[0]
                    # 如果北向资金开始中等幅度净买入时：
                    for cheat_date in new_north_data.loc[not_out_days:][:5].index:
                        north_in = (new_north_data.loc[cheat_date, '当日买入成交净额(人民币)'] > north_limit.loc[cheat_date, 'north_middle_in'])
                        market_up = self.bench_close.loc[cheat_date, ['SZZZ', 'CYB', 'CYBZ', 'wind_A', 'avg_A']] / \
                                    self.bench_close.loc[get_pre_trade_date(cheat_date), ['SZZZ', 'CYB', 'CYBZ', 'wind_A','avg_A']] - 1 >= 0.01
                        north_out = new_north_data.loc[cheat_date,'当日买入成交净额(人民币)'] < north_limit.loc[cheat_date, 'north_middle_out']
                        # 如果北向资金大幅净流入，但市场没有上涨，则直接转空
                        if (north_in & (market_up.sum() < 3)) | (north_out):
                            buy_date = cheat_date
                            norht_out_signal.loc[buy_date,'turn_short_position'] = -1
                            break
                        elif north_in & (market_up.sum() >= 3):
                            find_date = new_north_data.loc[cheat_date:,'当日买入成交净额(人民币)'][1:11] < north_median_out.loc[cheat_date:][1:11]
                            if find_date.sum() >0 :
                                buy_date = find_date[find_date == True].index[0]
                                norht_out_signal.loc[buy_date, 'turn_short_position'] = -1
                            break

                    if buy_date != None:
                        # 先至少看空5日
                        sell_date = min(get_pre_trade_date(buy_date, -1), end_date)
                        norht_out_signal.loc[buy_date, 'turn_short_position'] = -1
                        while (sell_date not in new_north_data.index) & (sell_date <= end_date):
                            norht_out_signal.loc[sell_date, 'turn_short_position'] = -1
                            sell_date = get_pre_trade_date(sell_date, -1)
                        # 如果5天后，北向资金出现连续3日不再净流出，或者出现单日大幅净流入 & 市场大涨，那么此时停止看空
                        no_out_days = (new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-5:] >= 0).sum()
                        while (no_out_days < 3) & (net_in_up.loc[sell_date] == False):
                            if sell_date == end_date:
                                break
                            norht_out_signal.loc[sell_date, 'turn_short_position'] = -1
                            sell_date = get_pre_trade_date(sell_date, -1)
                            while (sell_date not in new_north_data.index) & (sell_date <= end_date):
                                norht_out_signal.loc[sell_date, 'turn_short_position'] = -1
                                sell_date = get_pre_trade_date(sell_date, -1)
                            no_out_days = (new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-5:] >= 0).sum()

        market_turnout_position = pd.DataFrame(columns=['buy_date', 'sell_date', 'period_days', 'bench_pct', 'maxup','maxdown'])
        date, i = norht_out_signal.index[0], 0
        while date <= norht_out_signal.index[-1]:
            if norht_out_signal.loc[date,'turn_short_position'] == -1:
                buy_date = date
                sell_date = get_pre_trade_date(buy_date,-1)
                while norht_out_signal.loc[sell_date,'turn_short_position'] == -1:
                    sell_date = get_pre_trade_date(sell_date, -1)

                priod_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[buy_date, bench] - 1
                maxup = self.bench_close.loc[buy_date:sell_date, bench].max() / self.bench_close.loc[buy_date, bench] - 1
                maxdown = self.bench_close.loc[buy_date:sell_date, bench].min() / self.bench_close.loc[buy_date, bench] - 1

                market_turnout_position.loc[i] = buy_date, sell_date, get_trade_date_interval(sell_date,buy_date), priod_pct, maxup, maxdown
                i += 1
                date = sell_date
            else:
                date = get_pre_trade_date(date, -1)

        # 北向资金净流出期间，市场跟随下跌，我们关注市场的变化
        norht_out_signal['together_short_position'] = 0
        for date in norht_out_signal['sell_signal'][norht_out_signal['sell_signal'] == -1].index:
            if (norht_out_signal.loc[date, 'sell_signal'] == -1) & (continue_out.loc[date, 'continuous_out_days'] >= 3):
                north_in_period = self.bench_close.loc[:date, ['SZZZ', 'CYBZ', 'CYB', 'wind_A', 'avg_A']].iloc[-int(continue_out.loc[date, 'continuous_out_days']) - 1:]
                max_down = north_in_period.loc[date] / north_in_period.max() - 1
                max_up = north_in_period.loc[date] / north_in_period.min() - 1
                # 即市场位于左侧的时候
                if ((max_down < - 0.03) & (max_up < abs(max_down / 2))).sum() > 2:
                    # 如果北向资金发生大幅净流入，并且市场大幅上涨，那么看多，为1
                    not_out_days = (continue_out.loc[date:, 'continuous_out_signal'] == True) | (norht_out_signal.loc[date:, 'sell_signal'] == -1)
                    if (not_out_days == False).sum() == 0:
                        continue
                    not_out_days = not_out_days[not_out_days == False].index[0]

                    for cheat_date in new_north_data.loc[not_out_days:][:5].index:
                        north_in = (new_north_data.loc[cheat_date, '当日买入成交净额(人民币)'] > north_limit.loc[cheat_date, 'north_middle_in'])
                        north_out = new_north_data.loc[cheat_date, '当日买入成交净额(人民币)'] < north_limit.loc[cheat_date, 'north_middle_out']

                        north_in_days = (new_north_data.loc[not_out_days:cheat_date, '当日买入成交净额(人民币)'].iloc[-3:] > 0).sum()
                        # 此时，如果在北向资金停止净流出之后的5日内，北向资金出现了大幅度净流入
                        if (north_in & (bench_pct.loc[cheat_date].sum() >= 3)) | (north_in_days ==3):
                            buy_date = cheat_date
                            norht_out_signal.loc[buy_date,'together_short_position'] = 1
                            sell_date = min(get_pre_trade_date(buy_date, -1), end_date)
                            # 如果北向资金出现大幅净流出，则停止；或者未来10日后停止
                            while sell_date < min(get_pre_trade_date(buy_date, -10),end_date):
                                while (sell_date not in new_north_data.index) & (sell_date <= end_date):
                                    norht_out_signal.loc[sell_date, 'together_short_position'] = 1
                                    sell_date = get_pre_trade_date(sell_date, -1)
                                north_sell_out = new_north_data.loc[sell_date, '当日买入成交净额(人民币)'] < min(-30,north_limit.loc[sell_date, 'north_middle_out'])
                                north_sell_in = (new_north_data.loc[sell_date, '当日买入成交净额(人民币)'] > north_limit.loc[sell_date, 'north_middle_in'])

                                if (north_sell_out == False) & ((north_sell_in & (market_up.sum() < 3)) == False) & (continue_out.loc[sell_date, 'continuous_out_days']<5):
                                    norht_out_signal.loc[sell_date, 'together_short_position'] = 1
                                    sell_date = get_pre_trade_date(sell_date, -1)
                                else:
                                    break
                            break

                        # 如果出现了净流出，或者大幅净流入但市场没有上涨，则看空
                        elif (north_in & (market_up.sum() < 3)) | (north_out):
                            buy_date = cheat_date
                            norht_out_signal.loc[buy_date, 'together_short_position'] = -1
                            sell_date = min(get_pre_trade_date(buy_date, -1), end_date)
                            no_out_days = (new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-5:] >= 0).sum()
                            while (no_out_days < 3) & (net_in_up.loc[sell_date] == False):
                                if sell_date == end_date:
                                    break
                                norht_out_signal.loc[sell_date, 'together_short_position'] = -1
                                sell_date = get_pre_trade_date(sell_date, -1)
                                while (sell_date not in new_north_data.index) & (sell_date <= end_date):
                                    norht_out_signal.loc[sell_date, 'together_short_position'] = -1
                                    sell_date = get_pre_trade_date(sell_date, -1)
                                no_out_days = (new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-5:] >= 0).sum()
                            break

        market_leftout_position = pd.DataFrame(columns=['buy_date', 'sell_date', 'period_days', 'bench_pct', 'maxup','maxdown','signal'])
        date, i = norht_out_signal.index[0], 0
        while date <= norht_out_signal.index[-1]:
            if abs(norht_out_signal.loc[date,'together_short_position']) == 1:
                buy_date = date
                sell_date = get_pre_trade_date(buy_date, -1)
                while (abs(norht_out_signal.loc[sell_date, 'together_short_position']) == 1) & \
                        (norht_out_signal.loc[date,'together_short_position'] == norht_out_signal.loc[sell_date, 'together_short_position']):
                    sell_date = get_pre_trade_date(sell_date, -1)

                priod_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[buy_date, bench] - 1
                maxup = self.bench_close.loc[buy_date:sell_date, bench].max() / self.bench_close.loc[buy_date, bench] - 1
                maxdown = self.bench_close.loc[buy_date:sell_date, bench].min() / self.bench_close.loc[buy_date, bench] - 1

                market_leftout_position.loc[i] = buy_date, sell_date, get_trade_date_interval(sell_date, buy_date), \
                                                 priod_pct, maxup, maxdown, norht_out_signal.loc[date,'together_short_position']
                i += 1
                date = sell_date
            else:
                date = get_pre_trade_date(date, -1)


        norht_out_signal = norht_out_signal.fillna(0)
        norht_out_signal = pd.concat([continue_out[['continuous_out_days', 'count_days', 'big_out', 'middle_out']],norht_out_signal],axis=1)

        return norht_out_signal, market_out_position, market_turnout_position, market_leftout_position
    # 3、事件情况3：北向资金的逆向交易
    def north_reverse_transaction(self,start_date,end_date,bench = 'CYBZ'):
        # 北向资金的连续净流入和连续净流出的结果
        norht_in_signal, market_in_position = self.north_always_in(5,start_date,end_date,bench)
        norht_out_signal, market_out_position,market_turnout_position,market_leftout_position = self.north_always_out(5, start_date, end_date, bench)
        # 获取北向资金数据
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0].loc[get_pre_trade_date(start_date, 20):end_date]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        north_limit = self.north_big_in(get_pre_trade_date(start_date, 20), end_date)
        north_big_flow = ((new_north_data['当日买入成交净额(人民币)'] > north_limit['north_big_in']) * 1 + (new_north_data['当日买入成交净额(人民币)'] < north_limit['north_big_out']) * -1).loc[start_date:end_date]
        north_big_flow = north_big_flow.reindex(self.date_list).fillna(0)
        reverse_signal = pd.DataFrame(0, index=get_date_range(start_date, end_date),columns=['reverse_in', 'reverse_out'])
        # 市场涨跌幅
        bench_pct = self.bench_close.pct_change()[['SZZZ', 'CYB', 'CYBZ', 'wind_A', 'avg_A']].loc[start_date:end_date]
        bench_down = ((bench_pct < -0.005).sum(axis=1) >= 3) | ((bench_pct < -0.01).sum(axis=1) >= 2) | (((bench_pct < 0).sum(axis=1) == 5))
        bench_up = ((bench_pct > 0.005).sum(axis=1) >= 3) | ((bench_pct > 0.01).sum(axis=1) >= 3) | ((bench_pct > 0).sum(axis=1) == 5)

        bench_max_down = (self.bench_low / self.bench_close.shift(1) - 1)[['SZZZ', 'CYB', 'CYBZ', 'wind_A', 'avg_A']].loc[start_date:end_date]
        long_low_line = long_below_line(self.bench_open, self.bench_close, self.bench_high, self.bench_low).sum(axis=1)
        long_up_line = long_high_line(self.bench_open, self.bench_close, self.bench_high, self.bench_low).sum(axis=1)

        # 背离信号1-逆势净流入：市场下跌时，北向资金大幅净流入
        notbuy_position = (norht_in_signal['continuous_in_days'] >= 5) & (norht_in_signal['buy_signal'] == 0) # 但是要剔除：在资金连续净流入时，没有买入信号的日期
        reverse_signal['reverse_in'] = ((north_big_flow == 1) & (~bench_up) & (~((long_up_line>=2) & (~bench_down))) & (~notbuy_position)).loc[start_date:end_date]
        reverse_in_result = pd.DataFrame(columns=['date','sell_date', 'period', 'pct_change', 'max_up', 'max_down'])
        date, n = reverse_signal['reverse_in'].index[0], 0
        while date < reverse_signal['reverse_in'].index[-1]:
            while date not in reverse_signal['reverse_in'].index:
                date = get_pre_trade_date(date, -1)

            if reverse_signal['reverse_in'].loc[date] == True:
                future_period = get_date_range(date,min(get_pre_trade_date(date,-11),self.end_date))
                # 如果逆势净流入后，出现了大幅净流出，则停止
                if north_big_flow.loc[future_period[0]:future_period[-1]].min() == -1:
                    future_period = get_date_range(date, north_big_flow.loc[future_period][north_big_flow.loc[future_period]==-1].index[0])

                sell_date = future_period[-1]
                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[date, bench] - 1
                max_up = self.bench_close.loc[date:sell_date, bench].max() / self.bench_close.loc[date, bench] - 1
                max_down = self.bench_close.loc[date:sell_date, bench].min() / self.bench_close.loc[date, bench] - 1

                min_period = get_trade_date_interval(sell_date, date)

                reverse_in_result.loc[n] = date, sell_date, min_period, period_pct, max_up, max_down
                n += 1
                date = sell_date
            else:
                date = get_pre_trade_date(date, -1)

        # 背离信号2-逆势净流出：市场上涨时，北向资金大幅净流出
        # 并且满足：不能有下影线，有下影线表示市场日内已经对北向资金的净流出做出了充分的反应
        # 并且满足：不能是连续净流入，但是不看多的信号
        notsell_position = (norht_out_signal['continuous_out_days'] >= 5) & (norht_out_signal['sell_signal'] == 0)
        reverse_signal['reverse_out'] = ((north_big_flow == -1) & (~bench_down) & ((bench_max_down <= -0.01).sum(axis=1) < 3) & (~(long_low_line>=2)) & (~notsell_position)).loc[start_date:end_date]
        reverse_out_result = pd.DataFrame(columns=['date', 'sell_date', 'min_period', 'pct_change', 'max_up','max_down'])
        date, m = reverse_signal['reverse_out'].index[0], 0
        while date < reverse_signal['reverse_out'].index[-1]:
            while date not in reverse_signal['reverse_out'].index:
                date = get_pre_trade_date(date, -1)

            if reverse_signal['reverse_out'].loc[date] == True:
                future_period = get_date_range(date, min(get_pre_trade_date(date, -11), self.end_date))
                # 如果逆势净流出后，出现了大幅净流入，则停止
                if north_big_flow.loc[future_period[0]:future_period[-1]].max() == 1:
                    future_period = get_date_range(date, north_big_flow.loc[future_period][north_big_flow.loc[future_period] == 1].index[0])

                sell_date = future_period[-1]
                min_period = get_trade_date_interval(sell_date, date)
                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[date, bench] - 1
                max_up = self.bench_close.loc[date:sell_date, bench].max() / self.bench_close.loc[date, bench] - 1
                max_down = self.bench_close.loc[date:sell_date, bench].min() / self.bench_close.loc[date, bench] - 1

                reverse_out_result.loc[m] = date, sell_date, min_period, period_pct, max_up, max_down
                m += 1
                date = sell_date
            else:
                date = get_pre_trade_date(date, -1)

        reverse_signal['reverse_in'] = reverse_signal['reverse_in'].astype(int) * 1
        reverse_signal['reverse_out'] = reverse_signal['reverse_out'].astype(int) * -1

        return reverse_signal, reverse_in_result, reverse_out_result
    # 4、时间去情况4：北向资金的纠错
    def north_change_wrong(self, start_date, end_date, bench='CYBZ'):
        # 北向资金连续净流入，连续净流出之后的纠错
        norht_in_signal, market_in_position = self.north_always_in(5, start_date, end_date,bench)  # 北向资金连续净流入
        norht_out_signal, market_out_position, market_turnout_position, market_leftout_position = self.north_always_out(5, start_date,end_date,bench)  # 北向资金连续净流出

        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0].loc[get_pre_trade_date(start_date, 20):end_date]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        north_limit = self.north_big_in(get_pre_trade_date(start_date, 20), end_date)
        north_big_flow = ((new_north_data['当日买入成交净额(人民币)'] > north_limit['north_big_in']) * 1 + \
                          (new_north_data['当日买入成交净额(人民币)'] < north_limit['north_middle_out']) * -1).loc[start_date:end_date]
        north_middle_flow = ((new_north_data['当日买入成交净额(人民币)'] > north_limit['north_middle_in']) * 1).loc[start_date:end_date]

        north_big_flow = north_big_flow.reindex(self.date_list).fillna(0)

        north_wrong = pd.DataFrame(0,index=norht_in_signal.index,columns=['turn_wrong','turn_big_in','turn_position','big_position'])
        for date in north_big_flow.index:
            # 如果当天是大幅净流出，且过去5日出现了连续净流入
            if (north_big_flow.loc[date] == -1) & (norht_in_signal.loc[get_pre_trade_date(date, 5):date, 'buy_signal'].max() == 1):
                # 如果未来5天又出现了大幅净流入，则为True
                after_5days = get_pre_trade_date(date, -5)
                if north_big_flow.loc[date:after_5days].max() == 1:
                    buy_date = north_big_flow.loc[date:after_5days][north_big_flow.loc[date:after_5days] == 1].index[0]
                    north_wrong.loc[buy_date,'turn_wrong'] = 1
            # 如果当天是大幅净流入，且过去5日出现了连续净流出
            elif ((north_big_flow.loc[date] == 1) | (north_middle_flow.loc[get_pre_trade_date(date, 3):date].sum()>=2)) & \
                    (norht_out_signal.loc[get_pre_trade_date(date, 5):date, 'sell_signal'].min() == -1):
                # 如果未来5天又出现了大幅净流出，则为True
                after_5days = get_pre_trade_date(date, -5)
                if north_big_flow.loc[date:after_5days].min() == -1:
                    buy_date = north_big_flow.loc[date:after_5days][north_big_flow.loc[date:after_5days] == -1].index[0]
                    north_wrong.loc[buy_date, 'turn_wrong'] = -1

            # 如果当前大幅净流出，过去5日有大幅净流入
            if (north_big_flow.loc[date] == -1) & (north_big_flow.loc[get_pre_trade_date(date, 5):date].max() == 1):
                # 未来5日有大幅净流入
                if north_big_flow.loc[date: get_pre_trade_date(date, -5)].max() == 1:
                    begin_date = north_big_flow.loc[get_pre_trade_date(date, 5):date][
                        north_big_flow.loc[get_pre_trade_date(date, 5):date] == 1].index[0]
                    buy_date = north_big_flow.loc[date: get_pre_trade_date(date, -5)][
                        north_big_flow.loc[date: get_pre_trade_date(date, -5)] == 1].index[0]
                    if new_north_data.loc[begin_date:buy_date, '当日买入成交净额(人民币)'].sum() > 20:
                        north_wrong.loc[buy_date, 'turn_big_in'] = 1
            # 如果当前大幅净流出，过去5日有大幅净流入
            elif ((north_big_flow.loc[date] == 1) | (north_middle_flow.loc[get_pre_trade_date(date, 3):date].sum()>=2)) & (north_big_flow.loc[get_pre_trade_date(date, 5):date].min() == -1):
                # 未来5日有大幅净流入
                if north_big_flow.loc[date: get_pre_trade_date(date, -5)].min() == -1:
                    begin_date = north_big_flow.loc[get_pre_trade_date(date, 5):date][
                        north_big_flow.loc[get_pre_trade_date(date, 5):date] == -1].index[0]
                    buy_date = north_big_flow.loc[date: get_pre_trade_date(date, -5)][
                        north_big_flow.loc[date: get_pre_trade_date(date, -5)] == -1].index[0]
                    if new_north_data.loc[begin_date:buy_date, '当日买入成交净额(人民币)'].sum() < -20:
                        north_wrong.loc[buy_date, 'turn_big_in'] = -1

        # 开始测试结果
        turn_wrong_result = pd.DataFrame(columns=['buy_date','sell_date','position','period_pct','max_up','max_down'])
        date,i = north_wrong.index[0],0
        while date <= north_wrong.index[-1]:
            if date == end_date:
                break
            elif north_wrong.loc[date,'turn_wrong'] == 1:
                buy_date = date
                north_wrong.loc[buy_date, 'turn_position'] = 1
                sell_date = min(get_pre_trade_date(date,-1),end_date) # 获取第二天
                # 如果是买入，则没有中等幅度净卖出即可继续看空
                while ((north_big_flow.loc[sell_date] != -1) | (-new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'][new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)']<0].sum() < new_north_data.loc[buy_date,'当日买入成交净额(人民币)'] / 2)) \
                        & ((new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-3:]<0).sum() < 3):
                    north_wrong.loc[sell_date, 'turn_position'] = 1
                    if sell_date == end_date:
                        break
                    sell_date = get_pre_trade_date(sell_date, -1)  # 获取第二天

                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[date, bench] - 1
                max_up = self.bench_close.loc[date:sell_date, bench].max() / self.bench_close.loc[date, bench] - 1
                max_down = self.bench_close.loc[date:sell_date, bench].min() / self.bench_close.loc[date, bench] - 1

                turn_wrong_result.loc[i] = buy_date, sell_date, 1, period_pct, max_up, max_down
                i += 1
                date = sell_date

            elif north_wrong.loc[date,'turn_wrong'] == -1:
                buy_date = date
                north_wrong.loc[buy_date, 'turn_position'] = -1
                sell_date = min(get_pre_trade_date(buy_date, -1), end_date)  # 获取第二天
                # 如果是卖出，则没有中等幅度净卖出即可继续看空
                while ((north_big_flow.loc[sell_date] != 1) | \
                        (new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'][new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)']>0].sum() < -new_north_data.loc[buy_date,'当日买入成交净额(人民币)'] / 2)) \
                        & ((new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-3:]>0).sum() < 3):
                    north_wrong.loc[sell_date, 'turn_position'] = -1
                    if sell_date == end_date:
                        break
                    sell_date = get_pre_trade_date(sell_date, -1)  # 获取第二天
                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[date, bench] - 1
                max_up = self.bench_close.loc[date:sell_date, bench].max() / self.bench_close.loc[date, bench] - 1
                max_down = self.bench_close.loc[date:sell_date, bench].min() / self.bench_close.loc[date, bench] - 1

                turn_wrong_result.loc[i] = buy_date, sell_date, -1, period_pct, max_up, max_down
                i += 1
                date = sell_date

            else:
                date = get_pre_trade_date(date,-1)

        big_wrong_result = pd.DataFrame(columns=['buy_date','sell_date','position','period_pct','max_up','max_down'])
        date,i = north_wrong.index[0],0
        while date <= north_wrong.index[-1]:
            if date == end_date:
                break

            if north_wrong.loc[date,'turn_big_in'] == 1:
                buy_date = date
                north_wrong.loc[buy_date, 'big_position'] = 1
                sell_date = min(get_pre_trade_date(date,-1),end_date) # 获取第二天
                # 如果是买入，则没有中等幅度净卖出即可继续看空
                while ((north_big_flow.loc[sell_date] != -1) | (-new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'][new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)']<0].sum() < new_north_data.loc[buy_date,'当日买入成交净额(人民币)'] / 2)) \
                        & ((new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-3:]<0).sum() < 3):
                    north_wrong.loc[sell_date, 'big_position'] = 1
                    if sell_date == end_date:
                        break
                    sell_date = get_pre_trade_date(sell_date, -1)  # 获取第二天

                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[date, bench] - 1
                max_up = self.bench_close.loc[date:sell_date, bench].max() / self.bench_close.loc[date, bench] - 1
                max_down = self.bench_close.loc[date:sell_date, bench].min() / self.bench_close.loc[date, bench] - 1

                big_wrong_result.loc[i] = buy_date, sell_date, 1, period_pct, max_up, max_down
                i += 1
                date = sell_date

            elif north_wrong.loc[date,'turn_big_in'] == -1:
                buy_date = date
                north_wrong.loc[buy_date, 'big_position'] = -1
                sell_date = min(get_pre_trade_date(buy_date, -1), end_date)  # 获取第二天
                # 如果是卖出，则没有中等幅度净卖出即可继续看空
                while ((north_big_flow.loc[sell_date] != 1) | \
                        (new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'][new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)']>0].sum() < -new_north_data.loc[buy_date,'当日买入成交净额(人民币)'] / 2)) \
                        & ((new_north_data.loc[buy_date:sell_date, '当日买入成交净额(人民币)'].iloc[-3:]>0).sum() < 3):
                    north_wrong.loc[sell_date, 'big_position'] = -1
                    if sell_date == end_date:
                        break
                    sell_date = get_pre_trade_date(sell_date, -1)  # 获取第二天
                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[date, bench] - 1
                max_up = self.bench_close.loc[date:sell_date, bench].max() / self.bench_close.loc[date, bench] - 1
                max_down = self.bench_close.loc[date:sell_date, bench].min() / self.bench_close.loc[date, bench] - 1

                big_wrong_result.loc[i] = buy_date, sell_date, -1, period_pct, max_up, max_down
                i += 1
                date = sell_date

            else:
                date = get_pre_trade_date(date,-1)

        return north_wrong, turn_wrong_result, big_wrong_result
    # 5、事件情况5：北向资金和市场走势的背离
    def north_market_corr(self,start_date,end_date,bench = 'CYBZ'):
        # 先计算北水和市场指数走势的相关系数
        new_north_data = self.north_data[self.north_data['累计买入成交净额(人民币)'] != 0].loc[get_pre_trade_date(start_date, 20):end_date]  # 因为有很多交易日，北向资金不能交易，所以要剔除
        north_limit = self.north_big_in(get_pre_trade_date(start_date, 20), end_date)
        north_corr = rolling_corr(new_north_data[['累计买入成交净额(人民币)']], self.bench_close[[bench]].loc[new_north_data.index], window=10)['累计买入成交净额(人民币)']

        north_big_flow = ((new_north_data['当日买入成交净额(人民币)'] > north_limit['north_big_in']) * 1 + (new_north_data['当日买入成交净额(人民币)'] < north_limit['north_big_out']) * -1).loc[start_date:end_date]
        # 信号1：市场趋势和北向资金趋势相背离——北向资金大幅净流入，北向资金大幅净流出
        # 相关系数 ＜ -0.3时 ，北向资金大幅净流入；或者相关系数下降幅度超过0.5 & 当前相关系数＜0时，北向资金大幅净流入
        reverse_corr = ((north_corr < -0.3) & ((north_corr - north_corr.rolling(10).min() < 0.3))) | ((north_corr.rolling(5).max() - north_corr > 0.4) & (north_corr - north_corr.rolling(5).min() < 0.2) & (north_corr < 0.2))
        bench_pct = (self.bench_close / self.bench_close.rolling(10).max()-1)[['SZZZ','CYB','CYBZ','wind_A','avg_A']]
        #
        long_signal = (reverse_corr & (north_big_flow == 1) & ((bench_pct < -0.02).sum(axis=1) >=2)).loc[start_date:end_date]
        short_signal = (reverse_corr & (north_big_flow == -1) & ((bench_pct < -0.02).sum(axis=1) <=2)).loc[start_date:end_date]
        corr_signal = long_signal * 1 + short_signal * -1
        corr_signal = corr_signal.reindex(self.date_list).fillna(0).loc[start_date:end_date]
        corr_period = pd.DataFrame(corr_signal,columns=['signal'])
        # 北向资金的逆势净流入：看多————当北向资金走势和市场背离时，在北向资金放量/成交量占比较高时，北向资金出现了单日大幅净流入，认为是北向资金抄底的买点
        trade_result = pd.DataFrame(columns=['buy_date', 'sell_date', 'type','period', 'pct_change', 'max_up', 'max_down'])
        date, i = corr_signal.index[0], 0
        while date <= corr_signal.index[-1]:
            if date == end_date:
                break
            elif corr_signal.loc[date] == 1:
                buy_date = date
                corr_period.loc[buy_date,'period'] = 1
                # 信号1卖点：①北向资金和市场的相关性过高，②北向资金出现了大幅度净流出，也就是北向资金完成了抄底动作并开始拉升市场/纠错，那么就是卖点
                sell_date = min(get_pre_trade_date(date, -1),end_date)
                while sell_date not in north_corr.index:
                    corr_period.loc[sell_date, 'period'] = 1
                    sell_date = get_pre_trade_date(sell_date,-1)
                while (sell_date < get_pre_trade_date(date, -11)) & (north_corr.loc[sell_date] < 0.3) & (north_big_flow.loc[sell_date] != -1):
                    if sell_date == end_date:
                        break
                    corr_period.loc[sell_date, 'period'] = 1
                    sell_date = get_pre_trade_date(sell_date, -1)
                    while sell_date not in north_corr.index:
                        corr_period.loc[sell_date, 'period'] = 1
                        sell_date = get_pre_trade_date(sell_date, -1)

                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[buy_date, bench] - 1
                max_up = self.bench_close.loc[buy_date:sell_date, bench].max() / self.bench_close.loc[buy_date, bench] - 1
                max_down = self.bench_close.loc[buy_date:sell_date, bench].min() / self.bench_close.loc[buy_date, bench] - 1

                trade_result.loc[i] = buy_date, sell_date, 1, get_trade_date_interval(sell_date,date), period_pct, max_up, max_down
                i += 1
                date = sell_date
            elif corr_signal.loc[date] == -1:
                buy_date = date
                corr_period.loc[buy_date, 'period'] = -1
                # 卖点：①北向资金和市场的相关性过高，②北向资金出现了大幅度净流入，也就是北向资金完成了抄底动作并开始拉升市场/纠错，那么就是卖点
                sell_date = min(get_pre_trade_date(date, -1), end_date)
                while sell_date not in north_corr.index:
                    corr_period.loc[sell_date, 'period'] = -1
                    sell_date = get_pre_trade_date(sell_date,-1)

                while (sell_date < get_pre_trade_date(date, -11)) & (north_corr.loc[sell_date] < 0.3) & (north_big_flow.loc[sell_date] != 1):
                    if sell_date == end_date:
                        break
                    corr_period.loc[sell_date, 'period'] = -1
                    sell_date = get_pre_trade_date(sell_date, -1)
                    while sell_date not in north_corr.index:
                        corr_period.loc[sell_date, 'period'] = -1
                        sell_date = get_pre_trade_date(sell_date, -1)

                period_pct = self.bench_close.loc[sell_date, bench] / self.bench_close.loc[buy_date, bench] - 1
                max_up = self.bench_close.loc[buy_date:sell_date, bench].max() / self.bench_close.loc[
                    buy_date, bench] - 1
                max_down = self.bench_close.loc[buy_date:sell_date, bench].min() / self.bench_close.loc[
                    buy_date, bench] - 1

                trade_result.loc[i] = buy_date, sell_date, -1, get_trade_date_interval(sell_date,date), period_pct, max_up, max_down
                i += 1
                date = sell_date

            else:
                date = get_pre_trade_date(date,-1)

        return corr_period.fillna(0),trade_result


'''
start_date,end_date = 20160101,20221028
self = North_Money_Data(start_date,end_date)
save_start_date,save_end_date = 20190101,20221028
norht_in_signal, market_in_position = self.north_always_in(5,save_start_date,save_end_date,bench='CYBZ')  # 北向资金连续净流入
norht_out_signal, market_out_position,market_turnout_position,market_leftout_position = self.north_always_out(5,save_start_date,save_end_date,bench = 'CYBZ') # 北向资金连续净流出
reverse_signal, reverse_in_result, reverse_out_result = self.north_reverse_transaction(save_start_date,save_end_date,bench = 'CYBZ') # 北向资金的逆势操作
north_wrong, north_turn_in_position, north_turn_out_position = self.north_change_wrong(save_start_date,save_end_date,bench = 'CYBZ') # 北向资金的纠错
corr_signal,trade_result = self.north_market_corr(save_start_date,save_end_date,bench = 'CYBZ') # 北向资金和市场的背离


north_money_result = self.north_amt_weight()
trade_result.to_excel('C:/Users/86181/Desktop/北向资金.xlsx')
'''




