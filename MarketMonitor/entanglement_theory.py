import pandas as pd
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import datetime,time
from tqdm import tqdm
from dataApi import getData, stockList
from matplotlib import font_manager as fm, rcParams
font = fm.FontProperties(fname='E:/ABasicData/STKAITI.TTF')

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

# 函数2：画图
def draw_picture(FinallyResult,code,date,save_path= 'E:/',save_name='90%调整拐点'):
    pre_close = getData.get_daily_1factor('pre_close', date_list=[date], code_list=[code]).loc[date, code]
    FinallyResult[['high', 'low', 'open', 'close']] = FinallyResult[['high', 'low', 'open', 'close']] / pre_close - 1
    distance = max(abs(FinallyResult['high'].max()), abs(FinallyResult['low'].max()), 0.1)


    fig = plt.subplots(figsize=(30, 20))

    edge_colors = (FinallyResult['open'] > FinallyResult['close']).apply(lambda x: 'r' if x == False else 'g')

    ax1 = plt.subplot(3, 1, 1)
    ax1.bar(FinallyResult.index, (FinallyResult['close'] - FinallyResult['open']), 0.8, color=edge_colors,
            bottom=FinallyResult['open'], zorder=3)
    ax1.vlines(FinallyResult.index, FinallyResult['high'], FinallyResult['low'], color=edge_colors)
    for tl in ax1.get_xticklabels():
        tl.set_rotation(90)

    ax2 = ax1.twinx()
    ax2.bar(FinallyResult.index, abs(FinallyResult['拐点类型']),
            color=(FinallyResult['拐点类型'].apply(lambda x: 'r' if x > 0 else 'g').to_list()), width=0.2)
    ax2.set_ylim(0, 1)
    ax2.set_title('原始拐点', fontproperties=font, size=16)
    plt.margins(x=0)

    ax3 = plt.subplot(3, 1, 2)
    ax3.bar(FinallyResult.index, (FinallyResult['close'] - FinallyResult['open']), 0.8, color=edge_colors,
            bottom=FinallyResult['open'], zorder=3)
    ax3.vlines(FinallyResult.index, FinallyResult['high'], FinallyResult['low'], color=edge_colors)
    ax3.set_ylim(-distance, distance)
    for tl in ax3.get_xticklabels():
        tl.set_rotation(90)

    ax4 = ax3.twinx()
    ax4.bar(FinallyResult.index, abs(FinallyResult['第二类拐点类型']),
            color=(FinallyResult['第二类拐点类型'].apply(lambda x: 'r' if x > 0 else 'g').to_list()), width=0.2)
    ax4.set_ylim(0, 1)
    ax4.set_title('偏离度拐点', fontproperties=font, size=16)
    plt.margins(x=0)

    ax5 = plt.subplot(3, 1, 3)
    ax5.bar(FinallyResult.index, (FinallyResult['close'] - FinallyResult['open']), 0.8, color=edge_colors,
            bottom=FinallyResult['open'], zorder=3)
    ax5.vlines(FinallyResult.index, FinallyResult['high'], FinallyResult['low'], color=edge_colors)
    ax5.plot(yellow_pct.index, yellow_pct[code], color='y')
    ax5.plot(yellow_pct.index, yellow_pct['水平'], "--", color='black')
    ax5.set_ylim(-distance, distance)
    for tl in ax5.get_xticklabels():
        tl.set_rotation(90)

    ax6 = ax5.twinx()
    ax6.bar(FinallyResult.index, abs(FinallyResult['第三类拐点类型']),
            color=(FinallyResult['第三类拐点类型'].apply(lambda x: 'r' if x > 0 else 'g').to_list()), width=0.2)
    ax6.set_ylim(0, 1)
    ax6.set_title('因子拐点', fontproperties=font, size=16)
    plt.margins(x=0)
    if type(save_path) == str:
        plt.savefig(save_path + str(code) + '/'+ save_name + str(date) + '.jpg', dpi=200, bbox_inches='tight')
        send_file(save_path + str(code) + '/'+ save_name + str(date) + '.jpg',['015624'])
    plt.show()

# 缠论的构建
class entanglement_theory(object):
    def __init__(self, start_date, end_date,code_list=None):
        date_list = getData.get_date_range(start_date, end_date)
        self.start_date,self.end_date = date_list[0], date_list[-1]
        self.date_list =date_list
        # 个股日频数据
        pre_close = getData.get_daily_1factor('pre_close', date_list=date_list, code_list=code_list)
        open = getData.get_daily_1factor('open_badj', date_list=date_list, code_list=code_list)
        close = getData.get_daily_1factor('close_badj',date_list=date_list,code_list=code_list)
        high = getData.get_daily_1factor('high_badj', date_list=date_list, code_list=code_list)
        low = getData.get_daily_1factor('low_badj', date_list=date_list, code_list=code_list)

        self.open, self.close, self.high, self.low = open, close, high, low
        self.pre_close = pre_close
        # 指数的基础数据
        bench_close = getData.get_daily_1factor('close',date_list=date_list,type='bench')
        bench_open = getData.get_daily_1factor('open',date_list=date_list,type='bench')
        bench_high = getData.get_daily_1factor('high', date_list=date_list, type='bench')
        bench_low = getData.get_daily_1factor('low', date_list=date_list, type='bench')

        self.bench_close,self.bench_open,self.bench_high, self.bench_low = bench_close, bench_open, bench_high, bench_low

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

    FinallyResult.to_pickle('E:/FinallyResult.pkl')
    # 缠论第一步：分型，K线预处理
    def Kline_Type_Daily(self,code):
        # 用缠论的方式构造顶分型，和底分型
        Kline = pd.concat([self.high[code], self.low[code]], axis=1)
        Kline.columns = ['high', 'low']
        # 1、处理K线包含关系：默认第一天为起始点，建立第一天的情况
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
                    up = self.close.loc[start_time,code] >= self.open.loc[start_time,code]
                    down = self.close.loc[start_time,code] < self.open.loc[start_time,code]

                # 如果是上升，那么就取两者high和low的最大值；如果是下降，那么就取两者high和low的最小值
                if up == True:
                    price_change.iloc[-1]['high'] = max( price_change.iloc[-1]['high'],Kline.loc[now_time, 'high'])
                    price_change.iloc[-1]['low'] = max( price_change.iloc[-1]['low'],Kline.loc[now_time, 'low'])

                elif down == True:
                    price_change.iloc[-1]['high'] = min(price_change.iloc[-1]['high'], Kline.loc[now_time, 'high'])
                    price_change.iloc[-1]['low'] = min(price_change.iloc[-1]['low'], Kline.loc[now_time, 'low'])

        # 2、确定分型：1是顶分型，2是底分型
        index = 1
        while index < len(price_change)-1:
            today = price_change.index[index]
            yesterday, tomorrow, Up_Type, Down_Type =get_entanglement_type(price_change,today)
            # 顶分型与底分型之间，必须间隔1根K线
            if Up_Type == True:
                price_change.loc[[yesterday, today, tomorrow],'分型'] = 1
                # 先把时间往后推2天，看是不是又是一个顶分型
                index += 2
                while (Up_Type == True) & (index < len(price_change)-1):
                    # 先测试一下+2，如果不行就+3（即再+1）
                    after2_today = price_change.index[index]
                    after2_yesterday,after2_tomorrow,Up_Type,Down_Type = get_entanglement_type(price_change,after2_today)
                    # 如果仍然满足顶分型，且新的顶分型高点比原顶分型高点高，则保留后面的顶分型；否则后面的不记做顶分型
                    if Up_Type == True:
                        if price_change.loc[after2_today, 'high'] > price_change.loc[today,'high']:
                            price_change.loc[[after2_today, after2_yesterday, after2_tomorrow], '分型'] = 1

                            today = price_change.index[index]
                            index += 2
                        else:
                            Up_Type = False
                            index += 2

                    else: # 如果+2不行，那就+3试一下是否成立顶分型
                        index += 1
                        after3_today = price_change.index[index]
                        after3_yesterday, after3_tomorrow, Up_Type, Down_Type = get_entanglement_type(price_change,after3_today)
                        # 如果仍然满足顶分型，且新的顶分型高点比原顶分型高点高，则保留后面的顶分型；否则后面的不记做顶分型
                        if Up_Type == True:
                            if price_change.loc[after3_today, 'high'] > price_change.loc[today, 'high']:
                                price_change.loc[[after3_today, after3_yesterday, after3_tomorrow], '分型'] = 1

                                today = price_change.index[index]
                                index += 2
                            else:
                                Up_Type = False
                                index += 1
                        else:
                            index += 1

                #index += 2 # 把时间往后推4天（把时间往后推了4天）
            elif Down_Type == True:
                price_change.loc[[yesterday, today, tomorrow], '分型'] = -1
                # 先把时间往后推两天，看是不是又是一个底分型
                index += 2
                while (Down_Type == True) & (index < len(price_change)-1):
                    after2_today = price_change.index[index]
                    after2_yesterday, after2_tomorrow, Up_Type, Down_Type = get_entanglement_type(price_change,after2_today)
                    # 如果仍然满足底分型，且新的底分型低点比原底分型低点低，则保留后面的底分型；否则后面的不记做底分型
                    if Down_Type == True:
                        if price_change.loc[after2_today, 'low'] < price_change.loc[today,'low']:
                            price_change.loc[[after2_today, after2_yesterday, after2_tomorrow], '分型'] = -1

                            today = price_change.index[index]
                            index += 2
                        else:
                            Down_Type = False
                            index += 2
                    else:# 如果+2不行，那就+3试一下是否成立顶分型
                        index += 1
                        after3_today = price_change.index[index]
                        after3_yesterday, after3_tomorrow, Up_Type, Down_Type = get_entanglement_type(price_change,after3_today)
                        if Down_Type == True:
                            if price_change.loc[after3_today, 'low'] < price_change.loc[today, 'low']:
                                price_change.loc[[after3_today, after3_yesterday, after3_tomorrow], '分型'] = -1

                                today = price_change.index[index]
                                index += 2
                            else:
                                Down_Type = False
                                index += 1
                        else:
                            index += 1


                #index += 2  # 把时间往后推4天（把时间往后推了4天）
            else:
                index += 1  # 如果不是分型，把时间往后推1天

        price_change.to_pickle('/data/user/015624/Kline.pkl')


    # 拐点的筛选
    def Select_Point(self,Type_Change_Result,code, date):
        #Type_Change_Result = self.Kline_Type_Min(code, date, save_path='/data/user/015624/缠论拐点/')
        next_result = Type_Change_Result.copy()
        next_result['剔除原因'] = np.nan
        # 获取数据情况
        self.cal_other_data(date,code,sigma=2)
        # 剔除内容1：涨跌幅偏离过大的情况
        # Up_max区间，就不应该有买点（即拐点为1）；Down_max区间，就不该有卖点（即拐点为-1）
        next_result.loc[(Type_Change_Result['拐点类型'] == 1) & self.Up_max,['拐点类型']] = np.nan
        next_result.loc[(Type_Change_Result['拐点类型'] == 1) & self.Up_max, ['剔除原因']] = '涨幅偏离过大'
        next_result.loc[(Type_Change_Result['拐点类型'] == -1) & self.Down_max, ['拐点类型']] = np.nan
        next_result.loc[(Type_Change_Result['拐点类型'] == -1) & self.Down_max, ['剔除原因']] = '跌幅偏离过大'

        # 剔除内容2：偏离度筛选过大，和过小的情况
        # 偏离度连续5分钟过高时，只有卖点；偏离度连续5分钟过低时，只有买点
        Up_deviation = self.Up_deviation.rolling(5).sum() == 5 # 向上偏离过大，没有买点
        Down_deviation = self.Down_deviation.rolling(5).sum() == 5 # 向下偏离过大，没有卖点

        next_result.loc[(Type_Change_Result['拐点类型'] == 1) & Up_deviation,['拐点类型']] = np.nan
        next_result.loc[(Type_Change_Result['拐点类型'] == 1) & Up_deviation, ['剔除原因']] = '向上偏离过大'
        next_result.loc[(Type_Change_Result['拐点类型'] == -1) & Down_deviation, ['拐点类型']] = np.nan
        next_result.loc[(Type_Change_Result['拐点类型'] == -1) & Down_deviation, ['剔除原因']] = '向下偏离过大'
        # 向上偏离度过低时，只有买点；向下偏离度过低时，只有卖点
        # 连续3分钟向上偏离过低时，只有买点；连续3分钟向下偏离过低时，只有卖点
        Up_No_deviation = self.Up_No_deviation.rolling(3).sum() == 3
        Down_No_deviation = self.Down_No_deviation.rolling(3).sum() == 3

        next_result.loc[(Type_Change_Result['拐点类型'] == -1) & Up_No_deviation, ['拐点类型']] = np.nan
        next_result.loc[(Type_Change_Result['拐点类型'] == -1) & Up_No_deviation, ['剔除原因']] = '向上偏离过小'
        next_result.loc[(Type_Change_Result['拐点类型'] == 1) & Down_No_deviation, ['拐点类型']] = np.nan
        next_result.loc[(Type_Change_Result['拐点类型'] == 1) & Down_No_deviation, ['剔除原因']] = '向下偏离过小'

        return next_result
    # 拐点的添加
    def Append_Point(self,next_result,date,code,bid_amt_weight=2,bid_pct_weight=0.02):
        # 添加内容1：价格异动的情况
        next_result.loc[self.Up_turn[self.Up_turn==True].index,'拐点类型'] = 1
        next_result.loc[self.Up_turn[self.Up_turn == True].index, '分型'] = \
            next_result.loc[self.Up_turn[self.Up_turn == True].index, '分型'].apply(lambda x:x.append('上涨异动') if x==str else '上涨异动')

        next_result.loc[self.Down_turn[self.Down_turn == True].index, '拐点类型'] = -1
        next_result.loc[self.Down_turn[self.Down_turn == True].index, '分型'] = \
            next_result.loc[self.Down_turn[self.Down_turn == True].index, '分型'].apply(lambda x: x.append('下跌异动') if x == str else '下跌异动')

        # 添加内容2：集合竞价的异常情况
        up_abnormal_bid = (self.abnormal_bid_amt>=bid_amt_weight) | self.tick_continue_up | (self.bid_pct>=bid_pct_weight)
        down_abnormal_bid = self.tick_continue_down | (self.bid_pct<=-bid_pct_weight)

        #abnormal_bid = (self.abnormal_bid_amt>=bid_amt_weight) | (abs(self.bid_pct)>=2) | self.tick_continue_up | self.tick_continue_down
        if up_abnormal_bid == True:
            next_result.loc[925, '拐点类型'] = 1
        elif down_abnormal_bid == True:
            next_result.loc[925, '拐点类型'] = -1

        next_result.loc[925,['high','low','open','close']] = self.min_open.loc[date].loc[925,code]
        next_result.sort_index(inplace=True)

        return next_result

    # 得到最终结果
    def get_result(self,code,date,read_pkl=0,save_path='/data/user/015624/缠论拐点/'):
        Type_Change_Result = self.Kline_Type_Min(code, date, save_path)
        next_result = self.Select_Point(Type_Change_Result,code,date)
        next_result = self.Append_Point(next_result,date,code,bid_amt_weight=2, bid_pct_weight=0.02)
        # 读取一下异常点情况
        if read_pkl == 0:
            abnormal_result = pd.read_pickle(save_path + str(code) + '.pkl').loc[date]
        else:
            abnormal_result = pd.read_pickle(save_path + str(code) + '调整90.pkl').loc[date]

        next_result['最终拐点'] = ((next_result['拐点类型'] ==1) & abnormal_result['向上异常'])*1 + \
                              ((next_result['拐点类型'] == -1) & abnormal_result['向下异常'])*-1
        # 设置三份拐点：第一个是原始缠论的拐点；第二个是缠论通过涨跌幅、偏离度筛选和价格异常、竞价异常添加后的拐点
        # 第三个是再补充量能异常和盘口异常的拐点
        FinallyResult = Type_Change_Result.copy()
        FinallyResult.loc[925,['high','low','open','close']] = next_result.loc[925,['high','low','open','close']]
        FinallyResult.sort_index(inplace = True)

        FinallyResult['第二类拐点类型'] = next_result['拐点类型']
        FinallyResult['第二类分型'] = next_result['分型']
        FinallyResult['剔除原因'] = next_result['剔除原因']
        FinallyResult['第三类拐点类型'] = next_result['最终拐点']
        FinallyResult = pd.concat([FinallyResult,abnormal_result],axis=1)
        if type(save_path)  == str:
            FinallyResult.to_pickle('/data/user/015624/缠论拐点/'+str(code)+'/90%调整拐点'+str(date)+'.pkl')

        return FinallyResult


'''
mdp = xquant.marketdata.MarketData()
start_date,end_date,code = 20210101,20211231,300750
self = entanglement_theory(start_date,end_date,code)

# 测试个股分层情况的调整
for date in tqdm(self.date_list):
    FinallyResult = self.get_second_result(code, date,save_name='第二类调整再叠加')
    FinallyResult.index = pd.Series(FinallyResult.index).apply(lambda x: str(x).zfill(4))
    draw_picture(FinallyResult, code, date,save_name='股票分层拐点')

# 测试正常情况的结果
for date in tqdm(self.date_list):
    date = 20211224
    FinallyResult = self.get_result(code,date,read_pkl=1)
    FinallyResult.index = pd.Series(FinallyResult.index).apply(lambda x: str(x).zfill(4))
    draw_picture(FinallyResult,code,date,save_name='拐点结果')
'''

# 获取股票池
stock_pool = stockList.clean_stock_list().loc[20210104:20220628]
stock_pool = stock_pool[stock_pool.sum()[stock_pool.sum()>0].index]
for num in np.arange(800,len(stock_pool.columns),200):
    print(num)
    code_list = sorted(list(set(stock_pool.columns)))[num:num+200]
    # 准备数据
    start_date,end_date = 20160101,20220916
    code_list = None
    self = entanglement_theory(start_date,end_date,code_list)
    # 开始多进程
    p = Pool(20)
    for code in code_list:
        p.apply_async(self.get_two_result, args=(code,))
    p.close()
    p.join()

code= code_list[0]
self.get_two_result(code)

'''
read_path = '/data/user/015624/缠论拐点/'+str(code)+'/'
Result_Percent80 = pd.DataFrame()
for date in tqdm(self.date_list):
    today_result = pd.read_pickle(read_path + '90%调整拐点'+str(date)+'.pkl')
    today_result['date'] = date
    today_result = today_result.reset_index().set_index(['date','time'])

    Result_Percent80 = pd.concat([Result_Percent80,today_result])

Result_Percent80.to_pickle('/data/user/015624/缠论拐点/'+str(code)+'拐点结果-90%分位数.pkl')
'''

# 股票分层拐点3钟类型：
# 第一种：仅限于出现单边上涨，单边下跌，且偏离度大时，进行相应买卖点的剔除
# 第二种：在第一种的基础上，添加了当单边上涨单边下跌时，如果从最高点，或者最低点回撤和反弹低于振幅的1/10，则进行相应买卖点剔除

def draw_daily_picture(FinallyResult,bench_code,save_path= '/data/user/015624/缠论拐点/',save_name='90%调整拐点'):
    start_date, end_date = FinallyResult.index[0], FinallyResult.index[-1]
    date_list = get_date_list(start_date,end_date,delay=10)
    # 进行净值转化
    New_FinallyResult = pd.DataFrame(index=FinallyResult.index, columns=FinallyResult.columns)
    New_FinallyResult['open'] = FinallyResult['open']/FinallyResult['open'].iloc[0]
    New_FinallyResult['close'] = FinallyResult['close']/FinallyResult['open'] * New_FinallyResult['open']
    New_FinallyResult['high'] = FinallyResult['high'] / FinallyResult['open'] * New_FinallyResult['open']
    New_FinallyResult['low'] = FinallyResult['low'] / FinallyResult['open'] * New_FinallyResult['open']

    New_FinallyResult['分型'] = FinallyResult['分型']
    # 输出均线
    bench_10mean = New_FinallyResult['close'].rolling(10).mean()

    distance = max(abs(FinallyResult['high'].max()), abs(FinallyResult['low'].max()), 0.1) # 计算一下画图距离

    yellow_pct = bench_10mean.loc[start_date:end_date]
    yellow_pct['水平'] = 1
    fig = plt.subplots(figsize=(30, 20))

    edge_colors = (FinallyResult['open'] > FinallyResult['close']).apply(lambda x: 'r' if x == False else 'g')

    ax1 = plt.subplot(3, 1, 1)
    ax1.bar(FinallyResult.index, (FinallyResult['close'] - FinallyResult['open']), 0.8, color=edge_colors,
            bottom=FinallyResult['open'], zorder=3)
    ax1.vlines(FinallyResult.index, FinallyResult['high'], FinallyResult['low'], color=edge_colors)
    ax1.plot(yellow_pct.index, yellow_pct[code], color='y')
    ax1.plot(yellow_pct.index, yellow_pct['水平'], "--", color='black')
    ax1.set_ylim(-distance, distance)
    for tl in ax1.get_xticklabels():
        tl.set_rotation(90)

    ax2 = ax1.twinx()
    ax2.bar(FinallyResult.index, abs(FinallyResult['分型']),
            color=(FinallyResult['拐点类型'].apply(lambda x: 'r' if x > 0 else 'g').to_list()), width=0.2)
    ax2.set_ylim(0, 1)
    ax2.set_title('原始拐点', fontproperties=font, size=16)
    plt.margins(x=0)

    plt.margins(x=0)
    if type(save_path) == str:
        plt.savefig(save_path + '/'+ save_name + str(date) + '.jpg', dpi=200, bbox_inches='tight')
        send_file(save_path + '/'+ save_name + str(date) + '.jpg',['015624'])
    plt.show()

FinallyResult.to_pickle('/data/user/015624/Kline.pkl')
