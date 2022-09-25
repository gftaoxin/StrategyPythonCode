import datetime
import time as tm
import pandas as pd
from BasicData.local_path import *
from matplotlib import font_manager as fm, rcParams
import matplotlib.pyplot as plt
from matplotlib import ticker
from dataApi import getData,stockList,tradeDate


font=fm.FontProperties(fname='E:/ABasicData/STKAITI.TTF')

# 3、市场情绪：日频数据
def Point_Score(factor_name, Sentiment_Score, Sentiment):
    if factor_name=='投机情绪':
        Sentiment_Score['涨停数量'] = (Sentiment['当日涨停数量'] >= 90) * 10 + \
                                  ((Sentiment['当日涨停数量'] >= 45) & (Sentiment['当日涨停数量'] < 90)) * 7.5 + \
                                  ((Sentiment['当日涨停数量'] >= 25) & (Sentiment['当日涨停数量'] < 45)) * 5 + \
                                  ((Sentiment['当日涨停数量'] >= 10) & (Sentiment['当日涨停数量'] < 25)) * 2.5 + \
                                  ((Sentiment['当日涨停数量'] < 10)) * 0

        Sentiment_Score['连板数量'] = (Sentiment['当日连板数量'] >= 35) * 10 + \
                              ((Sentiment['当日连板数量'] >= 18) & (Sentiment['当日连板数量'] < 35)) * 7.5 + \
                              ((Sentiment['当日连板数量'] >= 9) & (Sentiment['当日连板数量'] < 18)) * 5 + \
                              ((Sentiment['当日连板数量'] >= 2) & (Sentiment['当日连板数量'] < 9)) * 2.5 + \
                              ((Sentiment['当日连板数量'] < 2)) * 0

        Sentiment_Score['炸板率'] = (Sentiment['当日炸板率'] < 0.2) * 10 + \
                                  ((Sentiment['当日炸板率'] > 0.2) & (Sentiment['当日炸板率'] <= 0.3)) * 7.5 + \
                                  ((Sentiment['当日炸板率'] > 0.3) & (Sentiment['当日炸板率'] <= 0.4)) * 5 + \
                                  ((Sentiment['当日炸板率'] > 0.4) & (Sentiment['当日炸板率'] <=0.5)) * 2.5 + \
                                  ((Sentiment['当日炸板率'] >0.5)) * 0

        Sentiment_Score['连板高度'] = (Sentiment['当日连板高度'] >= 5) * 10 + \
                              ((Sentiment['当日连板高度'] >= 3) & (Sentiment['当日连板高度'] <= 4)) * 6 + \
                              (Sentiment['当日连板高度'] == 2) * 3 + ((Sentiment['当日连板高度'] < 2)) * 0

    elif factor_name=='板块情绪':
        #########1、龙头股情绪#############
        Sentiment_Score['龙头股情绪'] = Sentiment['龙头股封板数量']*2 +\
                                   Sentiment['龙头股上涨6%数量']*1+\
                                   Sentiment['龙头股上涨3%数量']*0.5+\
                                   Sentiment['龙头股下跌数量']*-0.5+\
                                   Sentiment['龙头股下跌-3%数量']*-1+\
                                   Sentiment['龙头股下跌-6%数量']*-2+ \
                                   Sentiment['龙头股跌停数量'] * -2.5

        range_score = Sentiment['昨日龙头股数量'].apply(lambda x: 6 if x <= 3 else 10)
        ######如果最高板的封板高度
        range_score[Sentiment['当日连板高度'].rolling(242).max()>=5]=10

        Sentiment_Score['龙头股情绪'] = range_score * (Sentiment_Score['龙头股情绪'] + Sentiment['昨日龙头股数量'] * 2) / (Sentiment['昨日龙头股数量'] * 4)
        Sentiment_Score['龙头股情绪']=Sentiment_Score['龙头股情绪'].apply(lambda x:0 if x<0 else x)
        #########2、强势股情绪#############
        Sentiment_Score['强势股情绪'] = Sentiment['强势股封板数量']*2 +\
                                   Sentiment['强势股上涨6%数量']*1+\
                                   Sentiment['强势股上涨3%数量']*0.5+\
                                   Sentiment['强势股下跌数量']*-0.5+\
                                   Sentiment['强势股下跌-3%数量']*-1+\
                                   Sentiment['强势股下跌-6%数量']*-2+ \
                                   Sentiment['强势股跌停数量'] * -2.5

        range_score = Sentiment['昨日强势股数量'].apply(lambda x: 6 if x <= 7 else 10)
        Sentiment_Score['强势股情绪'] = range_score * (Sentiment_Score['强势股情绪'] + Sentiment['昨日强势股数量'] * 2) / (Sentiment['昨日强势股数量'] * 4)

    elif factor_name=='投机氛围':
        Sentiment_Score['涨停股溢价'] = \
            (Sentiment['昨日涨停股涨跌幅'] >= 0.06) * 10 + \
            ((Sentiment['昨日涨停股涨跌幅'] >= 0.05) & (Sentiment['昨日涨停股涨跌幅'] < 0.06)) * 7.5 + \
            ((Sentiment['昨日涨停股涨跌幅'] >= 0.035) & (Sentiment['昨日涨停股涨跌幅'] < 0.05)) * 5 + \
            ((Sentiment['昨日涨停股涨跌幅'] >= 0.01) & (Sentiment['昨日涨停股涨跌幅'] < 0.035)) * 2.5 + \
            ((Sentiment['昨日涨停股涨跌幅'] < 0.01)) * 0

        Sentiment_Score['炸板股溢价'] = \
            (Sentiment['昨日炸板股涨跌幅'] >= 0.03) * 10 + \
            ((Sentiment['昨日炸板股涨跌幅'] >= 0.02) & (Sentiment['昨日炸板股涨跌幅'] < 0.03)) * 7.5 + \
            ((Sentiment['昨日炸板股涨跌幅'] >= 0.01) & (Sentiment['昨日炸板股涨跌幅'] < 0.02)) * 5 + \
            ((Sentiment['昨日炸板股涨跌幅'] >= -0.01) & (Sentiment['昨日炸板股涨跌幅'] < 0.01)) * 2.5 + \
            ((Sentiment['昨日炸板股涨跌幅'] < -0.01)) * 0

        Sentiment_Score['追高股溢价'] = \
            (Sentiment['昨日追高股涨跌幅'] >= 0.045) * 10 + \
            ((Sentiment['昨日追高股涨跌幅'] >= 0.03) & (Sentiment['昨日追高股涨跌幅'] < 0.045)) * 7.5 + \
            ((Sentiment['昨日追高股涨跌幅'] >= 0.015) & (Sentiment['昨日追高股涨跌幅'] < 0.03)) * 5 + \
            ((Sentiment['昨日追高股涨跌幅'] >= 0) & (Sentiment['昨日追高股涨跌幅'] < 0.015)) * 2.5 + \
            (Sentiment['昨日追高股涨跌幅'] < 0) * 0

        Sentiment_Score['抄底股溢价'] = \
            (Sentiment['昨日抄底股涨跌幅'] >= 0.045) * 10 + \
            ((Sentiment['昨日抄底股涨跌幅'] >= 0.03) & (Sentiment['昨日抄底股涨跌幅'] < 0.045)) * 7.5 + \
            ((Sentiment['昨日抄底股涨跌幅'] >= 0.015) & (Sentiment['昨日抄底股涨跌幅'] < 0.03)) * 5 + \
            ((Sentiment['昨日抄底股涨跌幅'] >= 0) & (Sentiment['昨日抄底股涨跌幅'] < 0.015)) * 2.5 + \
            (Sentiment['昨日抄底股涨跌幅'] < 0) * 0

    return Sentiment_Score

class Daily_Market_Sentiment(object):
    def __init__(self,start_date,end_date):
        date_list = getData.get_date_range(start_date, end_date)

        self.start_date=date_list[0]
        self.end_date=date_list[-1]
        self.date_list = date_list

        pre_close = getData.get_daily_1factor('pre_close', date_list=date_list).dropna(how='all', axis=1).astype(float)
        open = getData.get_daily_1factor('open', date_list=date_list)[pre_close.columns].astype(float)
        high = getData.get_daily_1factor('high', date_list=date_list)[pre_close.columns].astype(float)
        low = getData.get_daily_1factor('low', date_list=date_list)[pre_close.columns].astype(float)
        close = getData.get_daily_1factor('close', date_list=date_list)[pre_close.columns].astype(float)
        amt = getData.get_daily_1factor('amt', date_list=date_list)[pre_close.columns].astype(float)
        turn = getData.get_daily_1factor('free_turn', date_list=date_list)[pre_close.columns].astype(float)
        Limit_Price = getData.get_daily_1factor('limit_up_price', date_list=date_list)[pre_close.columns].astype(float)
        Lowest_Price = getData.get_daily_1factor('limit_down_price', date_list=date_list)[pre_close.columns].astype(float)
        pct_chg = getData.get_daily_1factor('pct_chg', date_list=date_list)[pre_close.columns].astype(float)

        Limit_stock = (Limit_Price == close)  # 每日涨停个股
        Open_Board_stock = (Limit_Price > close) & (Limit_Price == high)  # 炸板个股


        Active_Stock = pd.read_hdf(base_address + 'FunctionData/'+'Active_Stock'+'.h5',key = 'Active_Stock').loc[start_date : end_date]
        stock_pool = pd.read_hdf(base_address + 'FunctionData/' + 'stock_pool' + '.h5', key='stock_pool').loc[start_date: end_date]
        All_Power_stock = pd.read_hdf(base_address + 'FunctionData/' + 'Power_stock' + '.h5', key='Power_stock').loc[start_date: end_date]
        Power_in_time = pd.read_hdf(base_address + 'FunctionData/' + 'Power_in_time' + '.h5', key='Power_in_time').loc[start_date: end_date]
        All_Power_stock = All_Power_stock & (Power_in_time <= 20)
        #############获取强势股，龙头股##################
        Dragon_Stock = pd.read_hdf(base_address + 'FunctionData/' + 'Dragon_Stock' + '.h5', key='Dragon_Stock').loc[start_date: end_date]

        # 强势个股为活跃板块中的强势个股,必须是非龙头股
        Power_stock = ((Active_Stock.rolling(5).max() == 1) & All_Power_stock) & ~Dragon_Stock.fillna(False)
        Power_stock = Power_stock  # 强势股

        Limit_High = pd.read_hdf(base_address + 'FunctionData/' + 'Limit_High' + '.h5', key='Limit_High').loc[start_date: end_date]

        ###抄底追高板块：近5日换手率位于市场前30% & 没有触板,且上涨的个股中，上涨最多的30只个股 ；反之亦然
        NoLimit_Stock = (high<Limit_Price) & (turn.rolling(5).mean().rank(axis=1, ascending=False,pct=True) < 0.3)

        buy_higher = ((close / pre_close - 1)[NoLimit_Stock & (close/pre_close-1>0)].rank(axis=1, ascending=False) <= 30)
        buy_lower = ((close / pre_close - 1)[NoLimit_Stock & (close/pre_close-1<0)].rank(axis=1, ascending=False) <= 30)

        self.Limit_High=Limit_High
        self.All_Power_stock = All_Power_stock & stock_pool  # 全市场强势股
        self.Power_stock = Power_stock & stock_pool  # 板块强势股
        self.Dragon_Stock = Dragon_Stock & stock_pool  # 龙头股
        self.Active_Stock = Active_Stock  #市场活跃股
        self.Limit_Price=Limit_Price
        self.close = close
        self.open=open
        self.high = high
        self.low=low
        self.pre_close = pre_close
        self.amt=amt
        self.Lowest_Price=Lowest_Price

        self.stock_pool = stock_pool
        self.Limit_stock = (Limit_stock & stock_pool)
        self.Open_Board_stock = Open_Board_stock & stock_pool

        self.buy_higher = buy_higher & stock_pool
        self.buy_lower = buy_lower & stock_pool
    #########板块情绪#########
    def Cal_Concpet(self, sentiment):
        stock_pct = (self.close / self.pre_close - 1)    ##收盘涨跌幅
        ########1、龙头股############
        Dragon_Stock = self.Dragon_Stock.shift(1)
        code_list = set(stock_pct.columns).intersection(set(self.Limit_stock.columns)).intersection(Dragon_Stock.columns)
        sentiment['昨日龙头股数量'] = Dragon_Stock.sum(axis=1)

        sentiment['龙头股封板数量'] = self.Limit_stock[code_list][Dragon_Stock[code_list] == True].sum(axis=1)
        sentiment['龙头股上涨6%数量'] =((stock_pct[code_list] >= 0.06) & self.Open_Board_stock[code_list])[Dragon_Stock[code_list] == True].sum(axis=1)
        sentiment['龙头股上涨3%数量']=((stock_pct[code_list] >= 0.03) & (stock_pct[code_list] < 0.06))[Dragon_Stock[code_list] == True].sum(axis=1)
        sentiment['龙头股下跌数量'] = ((stock_pct[code_list] >= -0.03) & (stock_pct[code_list] < 0))[Dragon_Stock[code_list] == True].sum(axis=1)
        sentiment['龙头股下跌-3%数量'] =((stock_pct[code_list] >= -0.06) & (stock_pct[code_list] < -0.03))[Dragon_Stock[code_list] == True].sum(axis=1)
        sentiment['龙头股下跌-6%数量'] =((stock_pct[code_list] >= -0.09) & (stock_pct[code_list] < -0.06))[Dragon_Stock[code_list] == True].sum(axis=1)
        sentiment['龙头股跌停数量'] = (stock_pct[code_list] <= -0.09)[Dragon_Stock[code_list] == True].sum(axis=1)

        sentiment['龙头股平均涨幅'] = stock_pct[code_list][Dragon_Stock[code_list] == True].mean(axis=1)
        ########2、强势股############
        Power_stock = self.Power_stock.shift(1)
        code_list = set(stock_pct.columns).intersection(set(self.Limit_stock.columns)).intersection(Power_stock.columns)
        sentiment['昨日强势股数量'] = Power_stock.sum(axis=1)

        sentiment['强势股封板数量'] = self.Limit_stock[code_list][Power_stock[code_list] == True].sum(axis=1)
        sentiment['强势股上涨6%数量'] = ((stock_pct[code_list] >= 0.06) & self.Open_Board_stock[code_list])[
            Power_stock[code_list] == True].sum(axis=1)
        sentiment['强势股上涨3%数量'] = ((stock_pct[code_list] >= 0.03) & (stock_pct[code_list] < 0.06))[
            Power_stock[code_list] == True].sum(axis=1)
        sentiment['强势股下跌数量'] = ((stock_pct[code_list] >= -0.03) & (stock_pct[code_list] < 0))[
            Power_stock[code_list] == True].sum(axis=1)
        sentiment['强势股下跌-3%数量'] = ((stock_pct[code_list] >= -0.06) & (stock_pct[code_list] < -0.03))[
            Power_stock[code_list] == True].sum(axis=1)
        sentiment['强势股下跌-6%数量'] = ((stock_pct[code_list] >= -0.09) & (stock_pct[code_list] < -0.06))[
            Power_stock[code_list] == True].sum(axis=1)
        sentiment['强势股跌停数量'] = (stock_pct[code_list] <= -0.09)[Power_stock[code_list] == True].sum(axis=1)

        sentiment['强势股平均涨幅'] = stock_pct[code_list][Power_stock[code_list] == True].mean(axis=1)

        return sentiment
    #######市场投机氛围##########
    def Cal_Market(self, sentiment):
        stock_pct = (self.close / self.pre_close - 1)    ##收盘涨跌幅
        #####1、昨日涨停个股溢价率##########
        Limit_stock = self.Limit_stock.shift(1)

        sentiment['昨日涨停股数量'] = Limit_stock.sum(axis=1)
        sentiment['昨日涨停股涨跌幅'] = stock_pct[Limit_stock == True].mean(axis=1)
        ######2、昨日炸板个股溢价率##########
        Open_Board_stock = self.Open_Board_stock.shift(1)

        sentiment['昨日炸板股数量'] = Open_Board_stock.sum(axis=1)
        sentiment['昨日炸板股涨跌幅'] = stock_pct[Open_Board_stock == True].mean(axis=1)
        ######3、昨日追高板块溢价率##########
        buy_higher = self.buy_higher.shift(1)

        sentiment['昨日追高股数量'] = buy_higher.sum(axis=1)
        sentiment['昨日追高股涨跌幅'] = stock_pct[buy_higher == True].mean(axis=1)
        ########4、昨日抄底板块溢价率########
        buy_lower = self.buy_lower.shift(1)

        sentiment['昨日抄底股数量'] = buy_lower.sum(axis=1)
        sentiment['昨日抄底股涨跌幅'] = stock_pct[buy_lower == True].mean(axis=1)

        return sentiment
        #######投机情绪#######
    #######投机情绪#######
    def Cal_Speculation(self, sentiment):
        sentiment['当日涨停数量'] = self.Limit_stock.sum(axis=1)
        sentiment['当日连板数量'] = (self.Limit_stock.rolling(2).sum()==2).sum(axis=1)
        sentiment['当日炸板率'] = self.Open_Board_stock.sum(axis=1) / (self.Open_Board_stock.sum(axis=1) + self.Limit_stock.sum(axis=1))

        sentiment['当日连板高度'] = (self.Limit_High.rolling(3).max().shift(1)*self.Limit_stock+self.Limit_stock).max(axis=1)
        return sentiment
    ######计算市场情绪########
    def Cal_sentiment(self):
        ############获取具体指标结果###########
        sentiment = pd.DataFrame(index=self.close.index,columns=['指数涨跌幅'])
        ####1、板块情绪#####
        sentiment=self.Cal_Concpet(sentiment)
        ####2、投机情绪####
        sentiment = self.Cal_Speculation(sentiment)
        ####3、市场投机氛围#####
        sentiment = self.Cal_Market(sentiment)

        self.Sentiment = sentiment
        ############获取情绪得分##################
        Sentiment_Score = pd.DataFrame(index=self.Sentiment.index, columns=['情绪得分', '投机情绪得分', '龙头情绪得分', '投机氛围得分'])
        Sentiment_Score = Point_Score('投机情绪', Sentiment_Score, self.Sentiment)
        Sentiment_Score = Point_Score('板块情绪', Sentiment_Score, self.Sentiment)
        Sentiment_Score = Point_Score('投机氛围', Sentiment_Score, self.Sentiment)

        self.Sentiment_Score = Sentiment_Score
        #############计算权重#################
        Weight = pd.DataFrame(index=self.date_list,columns=['涨停数量','连板数量','炸板率','连板高度',
                                                            '龙头股情绪','强势股情绪',
                                                            '涨停股溢价','炸板股溢价','追高股溢价','抄底股溢价'])

        Weight['涨停数量'] =  Weight['连板数量'] =0.15
        Weight['炸板率'] = 0.05
        Weight['连板高度'] = 0.05

        Weight['龙头股情绪']=0.2
        Weight['强势股情绪']=0.1

        Weight['涨停股溢价']=Weight['炸板股溢价']=0.1
        #########无追高板块，追高板块权重为0
        Weight['追高股溢价'] = 0.1
        Weight['追高股溢价'][self.buy_higher.sum(axis=1) == 0] = 0
        Weight['抄底股溢价'] = 0
        Weight['抄底股溢价'][Weight['追高股溢价']==0]=0.1

        #########如果无龙头股，权重给强势股；如果无强势股，权重给龙头股；如果既没有龙头股也没有强势股，则该部分得分为0
        Weight['龙头股情绪'][sentiment['昨日龙头股数量'] == 0] = 0
        Weight['强势股情绪'][sentiment['昨日龙头股数量'] == 0] = 0.25

        Weight['强势股情绪'][sentiment['昨日强势股数量'] == 0] = 0
        Weight['龙头股情绪'][sentiment['昨日强势股数量'] == 0] = 0.25

        Weight['强势股情绪'][sentiment['昨日强势股数量'] == 0][sentiment['昨日龙头股数量'] == 0] = 0
        Weight['龙头股情绪'][sentiment['昨日强势股数量'] == 0][sentiment['昨日龙头股数量'] == 0] = 0

        Weight['炸板率'][sentiment['当日涨停数量'] == 0] = 0

        ######市场情绪得分统计##########
        Score = (Sentiment_Score * Weight)

        self.Sentiment_Score['投机情绪得分']=Score[['涨停数量','连板数量','炸板率']].sum(axis=1)/0.35
        self.Sentiment_Score['龙头情绪得分']=Score[['龙头股情绪','连板高度']].sum(axis=1)/0.25
        self.Sentiment_Score['投机氛围得分']=Score[['强势股情绪','涨停股溢价','炸板股溢价','追高股溢价','抄底股溢价']].sum(axis=1)/0.4
        self.Sentiment_Score['情绪得分'] = Score.sum(axis=1)

        new_result = pd.DataFrame(index=self.date_list,columns=['情绪得分','投机情绪得分','龙头情绪得分','投机氛围得分','当日涨停数量','当日连板数量',
                                                                '当日炸板率','当日连板高度','龙头股平均涨幅','昨日涨停股涨跌幅','强势股平均涨幅'])
        new_result[['情绪得分','投机情绪得分','龙头情绪得分','投机氛围得分']] = self.Sentiment_Score[['情绪得分','投机情绪得分','龙头情绪得分','投机氛围得分']]

        new_result[['当日涨停数量','当日连板数量','当日炸板率','当日连板高度','龙头股平均涨幅','昨日涨停股涨跌幅','强势股平均涨幅']]= \
            self.Sentiment[['当日涨停数量','当日连板数量','当日炸板率','当日连板高度','龙头股平均涨幅','昨日涨停股涨跌幅','强势股平均涨幅']]
        self.new_result = new_result
    ####保存数据#######
    def save_Result(self,start_date,end_date,save_path='E:/ABasicData/'):
        writer = pd.ExcelWriter(save_path +str(end_date)+'_'+str(end_date)+'市场情绪分析.xlsx')
        round(self.Sentiment_Score.loc[start_date:end_date].T,2).to_excel(writer, sheet_name='日间情绪得分')
        round(self.Sentiment.loc[20220201:end_date],4).to_excel(writer, sheet_name='具体取值')
        writer.close()

def Cal_Sentiment_Picture(start_date,end_date):
    date_list = getData.get_date_range(start_date, end_date)

    self = Daily_Market_Sentiment(date_list[0], end_date=date_list[-1])
    self.Cal_sentiment()
    self.save_Result(start_date, end_date,save_path='E:/ABasicData/')  # 保存

    market_sentiment = pd.read_excel('E:/ABasicData/' + str(end_date) + '_' + str(end_date) + '市场情绪分析.xlsx',index_col=0).T
    market_sentiment = market_sentiment[['情绪得分', '投机情绪得分', '龙头情绪得分', '投机氛围得分']]
    market_sentiment.rename(columns={'情绪得分': '整体情绪得分', '投机情绪得分': '涨停投机情绪', '龙头情绪得分': '龙头情绪', '投机氛围得分': '赚钱效应'},inplace=True)
    market_sentiment = market_sentiment.rolling(5).mean().dropna()
    market_sentiment.index = market_sentiment.index.astype(str)

    fig = plt.subplots(figsize=(10, 10))
    ax1 = plt.subplot(211)
    ax1.plot(market_sentiment.index, market_sentiment['整体情绪得分'])
    # ax1.legend(['整体情绪得分'],prop=font,loc="upper left")
    for tl in ax1.get_xticklabels():
        tl.set_rotation(90)
    ax1.set_title('整体情绪得分', fontproperties=font, size=16)
    #ax1.axhline(y=0, ls=":", c="black")  # 添加水平直线
    plt.xticks([])
    #plt.ylim([3, 5.5])
    # ax2 = ax1.twinx()
    ax3 = plt.subplot(212)
    ax3.plot(market_sentiment.index, market_sentiment[['涨停投机情绪', '龙头情绪', '赚钱效应']].values)
    for tl in ax3.get_xticklabels():
        tl.set_rotation(90)
    ax3.set_title('各部分情绪得分', fontproperties=font, size=16)
    ax3.legend(['涨停投机情绪', '龙头情绪', '赚钱效应'], prop=font, loc="upper left")
    #ax3.axhline(y=0, ls=":", c="black")  # 添加水平直线

    for tl in ax3.get_xticklabels():
        tl.set_rotation(50)

    plt.tight_layout()
    plt.ylim([0, 9])
    plt.savefig('/E//ABasicData/市场情绪.png')
    #plt.show()


# 计算市场情绪
start_date, end_date = 20220101,20220914
Cal_Sentiment_Picture(start_date, end_date)





