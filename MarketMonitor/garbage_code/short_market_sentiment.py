
import matplotlib
from matplotlib import font_manager as fm, rcParams
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np
from dataApi import getData,stockList

# 计算涨跌停价格
def cal_limit_price(pre_close):
    Limit_price = round(pre_close * 1.1 + 0.0001, 2)
    stock_pool_688 = pd.Series(Limit_price.columns).apply(lambda x: x if (x // 1000 == 688) else np.nan).dropna().astype(int)
    stock_pool_688 = list(set(stock_pool_688))
    Limit_price.loc[20190722:, stock_pool_688] = round(pre_close.loc[20190722:, stock_pool_688] * 1.2 + 0.0001, 2)

    stock_pool_300 = pd.Series(Limit_price.columns).apply(
        lambda x: x if (x // 1000 == 300) else np.nan).dropna().astype(int)
    stock_pool_300 = list(set(stock_pool_300))
    Limit_price.loc[20200824:, stock_pool_300] = round(pre_close.loc[20200824:, stock_pool_300] * 1.2 + 0.0001, 2)

    Lowest_Price = round(pre_close * 0.9 + 0.0001, 2)
    stock_pool_688 = pd.Series(Lowest_Price.columns).apply(
        lambda x: x if (x // 1000 == 688) else np.nan).dropna().astype(int)
    stock_pool_688 = list(set(stock_pool_688))
    Lowest_Price.loc[20190722:, stock_pool_688] = round(pre_close.loc[20190722:, stock_pool_688] * 0.8 + 0.0001, 2)
    stock_pool_300 = pd.Series(Lowest_Price.columns).apply(
        lambda x: x if (x // 1000 == 300) else np.nan).dropna().astype(int)
    stock_pool_300 = list(set(stock_pool_300))
    Lowest_Price.loc[20200824:, stock_pool_300] = round(pre_close.loc[20200824:, stock_pool_300] * 0.8 + 0.0001, 2)

    return Limit_price,Lowest_Price

# 获取活跃的股票池
def get_active_pool(start_date,end_date):
    date_list = get_date_list(start_date, end_date, delay=240)
    close = getData.get_daily_1factor('close', date_list=date_list)
    high = getData.get_daily_1factor('high', date_list=date_list)
    low = getData.get_daily_1factor('low', date_list=date_list)
    pre_close = getData.get_daily_1factor('pre_close', date_list=date_list)
    # 全市场未开板新股：由于未上市前的状态是NaN，因此可以用上市后累计涨停状态是1 & 最高价=最低价 表示上市后未开板新股；ipo_date=1表示上剔除上市第一日#
    Limit_Price, Lowest_Price = cal_limit_price(pre_close)  # 涨跌停价
    Limit_stock = (Limit_Price[close.columns] == close) # 每日涨停个股

    ipo_date = getData.get_daily_1factor('live_days', date_list=date_list)
    ipo_one_board = ipo_date.copy()
    ipo_one_board[ipo_one_board == 1] = 0
    ipo_one_board.replace(0, np.nan, inplace=True)  # 把未上市之前的日期都变为0
    ipo_one_board[ipo_one_board > 0] = 1  # 上市之后的时间都标记为1
    ipo_one_board = (((ipo_one_board * Limit_stock).cumprod() == 1) & (high == low)) | (ipo_date == 1)
    stock_pool = stockList.clean_stock_list(no_ST=True, least_live_days=1, no_pause=True, least_recover_days=0).loc[start_date:end_date]
    stock_pool = (ipo_one_board == 0) & stock_pool
    # 去除ST个股
    return stock_pool.loc[start_date:end_date]
# 连板龙头股
def LimitDragon(start_date,end_date):
    date_list = get_date_list(start_date, end_date, delay=240)
    close_badj = getData.get_daily_1factor('close_badj', date_list=date_list)
    close = getData.get_daily_1factor('close', date_list=date_list)
    pre_close = getData.get_daily_1factor('pre_close', date_list=date_list)
    Limit_Price, Lowest_Price = cal_limit_price(pre_close)

    Limit_Stock = (Limit_Price[close.columns] == close)
    Strong_Stock = close_badj.pct_change(60).rank(pct=True,axis=1)
    # 10天内4板及以上，当前是市场领涨
    stock_choice = (Limit_Stock.rolling(10).sum()>=4) & (Strong_Stock>0.9)
    # 当前股价没有跌破20日均线
    line10 = close_badj.rolling(10).mean()

    LimitDragon = (stock_choice.rolling(20).max()==True) & (close_badj >line10)

    return LimitDragon
# 1、强势股
def Get_BenchMark_StockList(start_date,end_date):
    # （1）去掉ST个股：
    stock_list = stockList.clean_stock_list(no_ST=True, least_live_days=100, start_date=start_date, end_date=end_date)
    # （1）均线多头排列：5日线>20日线>30日线>60日线>240日线 & 收盘价>20日线
    date_list = get_date_list(start_date, end_date, delay=240)
    close_badj = getData.get_daily_1factor('close_badj',date_list=date_list)
    close = getData.get_daily_1factor('close', date_list=date_list)
    pre_close = getData.get_daily_1factor('pre_close',date_list=date_list)
    line5,line20,line60,line240 = close_badj.rolling(5).mean(),close_badj.rolling(20).mean(),\
                                         close_badj.rolling(60).mean(),close_badj.rolling(240).mean()

    BenchMarkStock = ((close_badj>line20) & (line5 > line20) & (line20 > line60) & (line60>line240))
    # （2）涨跌幅：近30日涨跌幅至少位于市场排名前90%，且自身涨跌幅＞30%
    Max_Pct = close_badj.pct_change(30,fill_method=None)
    Max_Pct_Rank = close_badj.pct_change(30,fill_method=None).rank(pct=True,axis=1)
    PctStock = ((Max_Pct_Rank>0.9) & (Max_Pct>0.3))
    # （3）涨停次数：涨停次数
    Limit_Price, Lowest_Price = cal_limit_price(pre_close)
    Limit_Time = (Limit_Price[close.columns] == close).rolling(30).sum() >= 3  # 30日内涨停次数≥3

    BenchMark_result = BenchMarkStock.loc[start_date:end_date] & PctStock.loc[start_date:end_date] \
                       & Limit_Time.loc[start_date:end_date] & stock_list.loc[start_date:end_date]

    return BenchMark_result
# 2、计算日间市场情绪
def MarketSentiment(start_date,end_date,rol_short=120,delay=30):
    date_list = getData.get_date_range(20150101, end_date) # 日期
    # 1、活跃股票池
    stock_pool = get_active_pool(start_date,end_date)
    stock_pool = stock_pool[stock_pool.sum()[stock_pool.sum()>0].index]
    code_list = set(stock_pool.columns)

    # 2、涨停数量，跌停数量（√）
    close = getData.get_daily_1factor('close',date_list=date_list)[code_list]
    high = getData.get_daily_1factor('high',date_list=date_list)[code_list]
    pre_close = getData.get_daily_1factor('pre_close',date_list=date_list)[code_list]

    Limit_Price,Lowest_Price = cal_limit_price(pre_close) # 涨跌停价

    Limit_stock = (Limit_Price == close) & stock_pool  # 每日涨停个股
    OpenBoard_stock = (Limit_Price > close) & (Limit_Price == high) & stock_pool   # 每日炸板个股
    Lowest_stock = (Lowest_Price == close) & stock_pool  # 每日跌停个股

    # 3、连板高度（√）
    limit_up_new = Limit_stock * stock_pool.replace(False, np.nan)
    Limit_High = limit_up_new.cumsum() - limit_up_new.cumsum()[limit_up_new == 0].ffill().fillna(0)
    Limit_High.fillna(0, inplace=True)

    # 4、涨跌幅
    Pct_change = getData.get_daily_1factor('pct_chg',date_list=date_list)[code_list]
    Pct_change = Pct_change[stock_pool.shift(1)==True]

    #############市场情绪####################
    # 1、涨停数量，涨停得分（√）
    Limit_Num = Limit_stock.sum(axis=1)
    Limit_Score = (ts_rank(Limit_Num, rol_day=rol_short) + ts_rank(Limit_Num))*5
    # 2、连板数量，连板得分（√）
    DoubleLimit_Num = (Limit_High>1).sum(axis=1)
    DoubleLimit_Score = (ts_rank(DoubleLimit_Num, rol_day=rol_short) + ts_rank(DoubleLimit_Num)) * 5
    # 3、连板高度，高度得分（√）
    LimitHigh_max = Limit_High.max(axis=1)
    LimitHigh_num = (Limit_High.T ==LimitHigh_max).sum()
    LimitHigh_Score = LimitHigh_max.apply(lambda x:10 if x>=6 else 8 if x==5 else 6 if x==4 else 4 if x==3 else 2 if x==2 else 0)
    LimitHigh_add = ((LimitHigh_num-1)*0.5).apply(lambda x:min(2,x))
    LimitHigh_add[LimitHigh_max==1]=0
    LimitHigh_Score = LimitHigh_Score+LimitHigh_add
    # 4、炸板率，炸板得分（√）
    Open_Weight = OpenBoard_stock.sum(axis=1)/(OpenBoard_stock.sum(axis=1)+Limit_stock.sum(axis=1))
    Open_Score =  Open_Weight.apply(lambda x:10 if x<=0.15 else 8 if ((x>0.15)&(x<=0.2)) else 6 if ((x>0.2)&(x<=0.3)) else 4 if ((x>0.3)&(x<=0.4))else 2 if ((x>0.4)&(x<0.5)) else 0)
    # 5、跌停板数量，跌停得分（√）
    Lowest_Num = Lowest_stock.sum(axis=1)
    Lowest_Score = 10-(ts_rank(Lowest_Num, rol_day=rol_short) + ts_rank(Lowest_Num)) * 5

    # 6、涨跌比：
    Pct_Score = (((Pct_change>0.5) & (Pct_change<5)).sum(axis=1) + ((Pct_change>5) & (Pct_change<7)).sum(axis=1)*1.5 + ((Pct_change>5) & (Pct_change>7)).sum(axis=1)*2)/Pct_change.count(axis=1)
    Pct_Score = Pct_Score.apply(lambda x:min(1,x))*10
    # 7、涨停溢价率(√）
    Limit_Premium = Pct_change[Limit_stock.shift(1)==True].mean(axis=1)
    LimitPremium_Score =  5+(Limit_Premium-2)
    LimitPremium_Score = LimitPremium_Score.apply(lambda x:max(min(10,x),0))
    # 8、炸板溢价率(√）
    Open_Premium = Pct_change[OpenBoard_stock.shift(1)==True].mean(axis=1)
    OpenPremium_Score = 5+Open_Premium
    OpenPremium_Score = OpenPremium_Score.apply(lambda x: max(min(10, x), 0))

    # 9、强势股溢价率
    Strong_Stock = Get_BenchMark_StockList(20150101, end_date)
    Strong_Premium = Pct_change[Strong_Stock.shift(1) == True].mean(axis=1)
    StrongPremium_Score = 5+Strong_Premium-0.5
    StrongPremium_Score = StrongPremium_Score.apply(lambda x: max(min(10, x), 0))
    # 10、强势股数量
    Strong_Num = Strong_Stock.sum(axis=1)
    Strong_Score = (ts_rank(Strong_Num, rol_day=rol_short) + ts_rank(Strong_Num))*5

    #####################合成市场情绪###########################
    MarketSentiment_Result = Limit_Score*0.2 + DoubleLimit_Score*0.1 + LimitHigh_Score*0.1 + Open_Score*0.1 + Lowest_Score*0.05 + \
             Pct_Score*0.1 + LimitPremium_Score*0.2 + OpenPremium_Score*0.1 + StrongPremium_Score*0.05
    MarketSentiment_Result.index = MarketSentiment_Result.index.astype(str)
    MarketSentiment_Result = pd.DataFrame(MarketSentiment_Result,columns=['市场情绪'])

    return MarketSentiment_Result.iloc[-delay:]
# 3、指数趋势
def MarketIndex_Trend(start_date,end_date):
    s = FactorData()
    date_list = get_date_list(start_date,end_date,delay=300) #'000016.SH','000300.SH','000905.SH'
    IndexClose = s.get_factor_value('WIND_AIndexEODPrices',factors=['S_INFO_WINDCODE', 'TRADE_DT','S_DQ_CLOSE'],
                             S_INFO_WINDCODE=['000001.SH','399006.SZ','399001.SZ'],trade_dt=['>='+str(date_list[0]),'<='+str(date_list[-1])])
    IndexClose = IndexClose.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')

    Line5,Line10,Line20 = IndexClose.rolling(5).mean(),IndexClose.rolling(10).mean(),IndexClose.rolling(20).mean()
    Line60,Line240 = IndexClose.rolling(60).mean(),IndexClose.rolling(240).mean()
    # 指数每上穿一根均线，+1分
    Index_Trend = (IndexClose>Line5)*1 + (IndexClose>Line10)*1+ (IndexClose>Line20)*1 + (IndexClose>Line60)*1 + (IndexClose>Line240)*1
    Index_Trend.rename(columns={'000001.SH':'上证综指','399001.SZ': '深证成指','399006.SZ': '创业板指数'},inplace=True)
    Index_Trend.index = Index_Trend.index.astype(int)

    # （2）一级行业趋势
    code_dict = {
        '801020.SI': '采掘',
        '801040.SI': '钢铁',
        '801050.SI': '有色金属',
        '801030.SI': '化工',
        '801710.SI': '建筑材料',
        '801170.SI': '交通运输',
        '801740.SI': '国防军工',
        '801890.SI': '机械设备',
        '801730.SI': '电气设备',
        '801720.SI': '建筑装饰',
        '801010.SI': '农林牧渔',
        '801140.SI': '轻工制造',
        '801120.SI': '食品饮料',
        '801130.SI': '纺织服装',
        '801110.SI': '家用电器',
        '801880.SI': '汽车',
        '801200.SI': '商业贸易',
        '801210.SI': '休闲服务',
        '801750.SI': '计算机',
        '801080.SI': '电子',
        '801770.SI': '通信',
        '801760.SI': '传媒',
        '801780.SI': '银行',
        '801790.SI': '非银金融',
        '801180.SI': '房地产',
        '801150.SI': '医药生物',
        '801160.SI': '公用事业',
        '801230.SI': '综合',
    }
    SW_list = list(code_dict.keys())
    SW_Close = s.get_factor_value('WIND_ASWSIndexEOD',factors=['S_INFO_WINDCODE', 'TRADE_DT','S_DQ_CLOSE'],S_INFO_WINDCODE=SW_list,trade_dt=['>='+str(date_list[0]),'<='+str(date_list[-1])])
    SW_Close = SW_Close.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')

    SWLine5,SWLine10,SWLine20 = SW_Close.rolling(5).mean(),SW_Close.rolling(10).mean(),SW_Close.rolling(20).mean()
    SWLine60,SWLine240 = SW_Close.rolling(60).mean(),SW_Close.rolling(240).mean()

    SW_Trend = (SW_Close>SWLine5)*1 + (SW_Close>SWLine10)*1+ (SW_Close>SWLine20)*1 + (SW_Close>SWLine60)*1 + (SW_Close>SWLine240)*1
    SW_Trend.rename(columns=code_dict,inplace=True)
    SW_Trend.index = SW_Trend.index.astype(int)

    # （3）总的结果
    Trend_Result = pd.DataFrame(index=get_date_list(start_date,end_date,delay=0),columns=['指数趋势','行业趋势','市场总趋势'])
    Trend_Result['指数趋势'] = Index_Trend.mean(axis=1)*2
    Trend_Result['行业趋势'] =  SW_Trend.mean(axis=1)*2
    Trend_Result['市场总趋势'] = Index_Trend.mean(axis=1) + SW_Trend.mean(axis=1)

    return round(Trend_Result,2)

#### 整体监控指标 ####
class ShortTrade_Sentiment(object):
    def __init__(self, start_date, end_date):
        date_list = get_date_list(start_date, end_date,delay=0)
        start_date,end_date = date_list[0],date_list[-1]
        self.start_date = start_date
        self.end_date = end_date
        self.date_list = date_list
        # 1、每日的股票列表
        stock_pool = get_active_pool(start_date,end_date)
        stock_pool = stock_pool[stock_pool.sum()[stock_pool.sum() > 0].index]
        code_list = set(stock_pool.columns)
        # 2、日频：个股量价信息
        stockpct = getData.get_daily_1factor('pct_chg', date_list=date_list)[code_list]

        pre_close = getData.get_daily_1factor('pre_close', date_list=date_list)[code_list]
        close = getData.get_daily_1factor('close', date_list=date_list)[code_list]
        high = getData.get_daily_1factor('high', date_list=date_list)[code_list]
        low = getData.get_daily_1factor('low', date_list=date_list)[code_list]
        vwap = getData.get_daily_1factor('vwap', date_list=date_list)[code_list]

        amt = getData.get_daily_1factor('amt', date_list=date_list)[code_list]
        turn = getData.get_daily_1factor('free_turn', date_list=date_list)[code_list]

        self.stockpct =  stockpct[stock_pool==True].dropna(how='all',axis=1)
        self.pre_close = pre_close
        self.close = close
        self.high = high
        self.low = low
        self.amt = amt
        self.turn = turn
        self.vwap = vwap

        # 3、日频：涨停，跌停状态；连板状态，连板高度，炸板率
        s = FactorData()
        StockResult = s.get_factor_value('Basic_factor', stock=[], mddate=[str(x) for x in date_list], factor_names=['mdc_maxpx','mdc_minpx'])
        Limit_Price = StockResult['mdc_maxpx'].unstack().dropna(how='all', axis=1)
        Lowest_Price = StockResult['mdc_minpx'].unstack().dropna(how='all', axis=1)
        #code_list = set(close.columns).intersection(set(Limit_Price.columns))
        Limit_Price.index, Limit_Price.columns = Limit_Price.index.astype(int),pd.Series(Limit_Price.columns).apply(lambda x: stockList.trans_windcode2int(x))
        Lowest_Price.index,Lowest_Price.columns =Lowest_Price.index.astype(int),pd.Series(Lowest_Price.columns).apply(lambda x: stockList.trans_windcode2int(x))

        Limit_stock = (Limit_Price[code_list] == close) & stock_pool  # 每日涨停个股
        OpenBoard_stock = (Limit_Price[code_list] > close) & (Limit_Price[code_list] == high) & stock_pool  # 炸板个股
        Lowest_stock = (Lowest_Price[code_list] == close)  & stock_pool # 每日跌停个股
        Limit_High = ContinuousTrueTime(Limit_stock) #连板高度

        # 获取强势股 和 龙头股
        Strong_Stock = Get_BenchMark_StockList(start_date,end_date)
        Limit_Dragon = LimitDragon(start_date,end_date)

        dragon_stock_old = ConceptApi.get_basic_values('Dragon_Stock', start_date=20150101, end_date=end_date)
        dragon_stock_new = ConceptApi.get_basic_values('Dragon_Stock', start_date=start_date, end_date=end_date,read_path='/data/group/800442/800319/Temporary_Data/RawData/BasicData/')
        Dragon_Stock = pd.concat([dragon_stock_old,dragon_stock_new])
        Dragon_Stock = Dragon_Stock.groupby(Dragon_Stock.index).last()

        self.Limit_Price,self.Lowest_Price = Limit_Price,Lowest_Price
        self.Limit_stock = Limit_stock
        self.OpenBoard_stock = OpenBoard_stock
        self.Lowest_stock = Lowest_stock
        self.Limit_High = Limit_High[stock_pool==True]
        self.Strong_Stock = Strong_Stock
        self.Dragon_Stock = Dragon_Stock
        self.Limit_Dragon = Limit_Dragon
        # 4、日内：当日的分钟数据，分钟最高价，分钟收盘价
        min_close = getData.get_minute_1factor('close',start_datetime=end_date,end_datetime=end_date,code_list=code_list).loc[end_date]
        min_high= getData.get_minute_1factor('high',start_datetime=end_date,end_datetime=end_date,code_list=code_list).loc[end_date]
        min_close.index,min_close.columns =  min_close.index.astype(str),pd.Series(min_close.columns).apply(lambda x:stockList.trans_int2windcode(x))
        min_high.index, min_high.columns = min_high.index.astype(str), pd.Series(min_high.columns).apply(lambda x: stockList.trans_int2windcode(x))

        #min_close,min_high = pd.DataFrame(columns=code_list),pd.DataFrame(columns=code_list)
        #for code in tqdm(close.columns):
        #    Result = ma.getMDSecurityKLineDataFrame(code, str(end_date) + '091500', str(end_date) + '150000', 10,20).set_index('MDTime')
        #    min_close[code],min_high[code] = Result['ClosePx'],Result['HighPx']

        self.min_close = min_close
        self.min_high = min_high
        self.stock_pool = stock_pool

    # 1、涨跌比：[,-7%],(-7%:-5%],(-5%:-3%],(-3%:-1%),[-1%:1%],(1%:3%),[3%:5%),[5%:7%),[7%:]
    def UpDownWeight(self,end_date):
        # 市场整体情况
        UpDown_Result = pd.DataFrame(index=['≥7%', '7%～5%', '5%～3%', '3%～1%', '1%～-1%', '-1%～-3%', '-3%～-5%', '-5%～-7%', '≤-7%'], columns=self.date_list)
        UpDown_Result.loc['≥7%'] = (self.stockpct>= 7).sum(axis=1)
        UpDown_Result.loc['7%～5%'] = ((self.stockpct >= 5) & (self.stockpct < 7)).sum(axis=1)
        UpDown_Result.loc['5%～3%'] = ((self.stockpct >= 3) & (self.stockpct < 5)).sum(axis=1)
        UpDown_Result.loc['3%～1%'] = ((self.stockpct > 1) & (self.stockpct < 3)).sum(axis=1)
        UpDown_Result.loc['1%～-1%'] = ((self.stockpct >= -1) & (self.stockpct <= 1)).sum(axis=1)
        UpDown_Result.loc['-1%～-3%'] = ((self.stockpct > -3) & (self.stockpct < -1)).sum(axis=1)
        UpDown_Result.loc['-3%～-5%'] = ((self.stockpct > -5) & (self.stockpct <= -3)).sum(axis=1)
        UpDown_Result.loc['-5%～-7%'] = ((self.stockpct > -7) & (self.stockpct <= -5)).sum(axis=1)
        UpDown_Result.loc['≤-7%'] = (self.stockpct <= -7).sum(axis=1)
        # 昨日涨停个股情况
        LimitPct = self.stockpct[self.Limit_stock.shift(1)==True]
        Limit_Result = pd.DataFrame(index=['≥7%', '7%～5%', '5%～3%', '3%～1%', '1%～-1%', '-1%～-3%', '-3%～-5%', '-5%～-7%', '≤-7%'],columns=self.date_list)
        Limit_Result.loc['≥7%'] = (LimitPct >= 7).sum(axis=1)
        Limit_Result.loc['7%～5%'] = ((LimitPct >= 5) & (LimitPct < 7)).sum(axis=1)
        Limit_Result.loc['5%～3%'] = ((LimitPct>= 3) & (LimitPct < 5)).sum(axis=1)
        Limit_Result.loc['3%～1%'] = ((LimitPct > 1) & (LimitPct < 3)).sum(axis=1)
        Limit_Result.loc['1%～-1%'] = ((LimitPct >= -1) & (LimitPct <= 1)).sum(axis=1)
        Limit_Result.loc['-1%～-3%'] = ((LimitPct > -3) & (LimitPct < -1)).sum(axis=1)
        Limit_Result.loc['-3%～-5%'] = ((LimitPct > -5) & (LimitPct <= -3)).sum(axis=1)
        Limit_Result.loc['-5%～-7%'] = ((LimitPct > -7) & (LimitPct <= -5)).sum(axis=1)
        Limit_Result.loc['≤-7%'] = (LimitPct <= -7).sum(axis=1)

        WeightPrint = pd.DataFrame(index=UpDown_Result.index,columns=['市场整体情况','涨停溢价率情况'])
        WeightPrint['市场整体情况'] = UpDown_Result[end_date]
        WeightPrint['涨停溢价率情况'] = Limit_Result[end_date]

        WeightPrint = WeightPrint.reset_index()
        font = fm.FontProperties(fname='/data/user/015624/STKAITI.TTF')
        fig = plt.subplots(figsize=(30, 10))
        ax1 = plt.subplot(121)
        ax1.bar(WeightPrint.index, WeightPrint['市场整体情况'],
                color=['r', 'r', 'orangered', 'orangered', 'gold', 'limegreen', 'limegreen', 'g', 'g'])
        ax1.legend(prop=font, loc="upper left")

        ax1.set_title(str(end_date) + '日内涨跌比情况', fontproperties=font, size=28)

        plt.xticks(WeightPrint.index, WeightPrint['index'], fontproperties=font, size=18)
        plt.yticks(fontproperties=font, size=18)
        ax3 = plt.subplot(122)
        ax3.bar(WeightPrint.index, WeightPrint['涨停溢价率情况'],
                color=['r', 'r', 'orangered', 'orangered', 'gold', 'limegreen', 'limegreen', 'g', 'g'])
        ax3.set_title(str(end_date) + '昨日涨停股涨跌情况', fontproperties=font, size=28)
        plt.xticks(WeightPrint.index, WeightPrint['index'], fontproperties=font, size=18)
        plt.yticks(fontproperties=font, size=18)

        plt.savefig('/data/user/015624/市场情绪/日内涨跌比.png')
        send_file('/data/user/015624/市场情绪/日内涨跌比.png')
        plt.show()

        return UpDown_Result,Limit_Result

    # 2、日内涨停情况：涨停数量，非一字板涨停数量，跌停数量
    def LimitState_Inday(self,date):
        Limit_Price = self.Limit_Price.loc[date]
        Limit_Price.index = pd.Series(Limit_Price.index).apply(lambda x: stockList.trans_int2windcode(x))
        Lowest_Price = self.Lowest_Price.loc[date]
        Lowest_Price.index = pd.Series(Lowest_Price.index).apply(lambda x: stockList.trans_int2windcode(x))
        stock_pool = self.stock_pool.loc[date][self.stock_pool.loc[date]==True].index.to_list()
        stock_pool =[stockList.trans_int2windcode(x) for x in stock_pool]

        LimitNum = (self.min_close[stock_pool] == Limit_Price[stock_pool]).sum(axis=1)
        NotOneWord_LimitNum = ((self.min_close[stock_pool] == Limit_Price[stock_pool]) & (self.min_close.cummin()[stock_pool]<Limit_Price[stock_pool])).sum(axis=1)
        DownNum = (self.min_close[stock_pool] == Lowest_Price[stock_pool]).sum(axis=1)

        # 连板情况：昨日涨停且今日涨停
        yesterday =self.date_list[self.date_list.index(date)-1]
        limit_stock_list = set(self.Limit_stock.loc[yesterday][self.Limit_stock.loc[yesterday]==True].index)
        limit_stock_list = [stockList.trans_int2windcode(x) for x in limit_stock_list]
        DoubleLimitNum = (self.min_close[limit_stock_list] == Limit_Price[limit_stock_list]).sum(axis=1)

        LimitState = pd.DataFrame(index=LimitNum.index,columns=['涨停数量','非一字涨停数量','连板数量','跌停数量'])
        LimitState['涨停数量'] = LimitNum
        LimitState['非一字涨停数量'] = NotOneWord_LimitNum
        LimitState['连板数量'] = DoubleLimitNum
        LimitState['跌停数量'] = DownNum
        LimitState.index = pd.Series(LimitState.index).apply(lambda x: x.rjust(4, '0'))

        def format_date(x, pos=None):
            if x < 0 or x > len(LimitState.index) - 1:
                return ''
            return LimitState.index[int(x)][:8]

        font = fm.FontProperties(fname='/data/user/015624/STKAITI.TTF')
        fig = plt.subplots(figsize=(30, 10))
        ax1 = plt.subplot(111)
        ax1.plot(LimitState.index, LimitState[['涨停数量', '非一字涨停数量','连板数量', '跌停数量']])
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax1.legend(['涨停数量', '非一字涨停数量','连板数量', '跌停数量'], prop=font, loc='upper left', fontsize=50)
        ax1.set_title(str(date)+'日内涨停情况', fontproperties=font, size=24)
        plt.grid(axis='y')
        plt.savefig('/data/user/015624/市场情绪/日内涨停情况.png')
        send_file('/data/user/015624/市场情绪/日内涨停情况.png')

    # 3、涨停数量，跌停数量，连板数量，连板高度，炸板率，溢价率
    def LimitState_Daily(self):
        LimitState = pd.DataFrame(columns=['涨停数量','跌停数量','连板数量','连板高度','炸板率','涨停溢价率','龙头股溢价率','强势股溢价率'])
        LimitState['涨停数量'] = self.Limit_stock.sum(axis=1)
        LimitState['跌停数量'] = self.Lowest_stock.sum(axis=1)
        LimitState['连板数量'] = (self.Limit_High>1).sum(axis=1)
        LimitState['连板高度'] = self.Limit_High.max(axis=1).astype(int)
        LimitState['炸板率'] = round(self.OpenBoard_stock.sum(axis=1)/(self.OpenBoard_stock.sum(axis=1)+self.Limit_stock.sum(axis=1))*100,2).astype(str)+'%'
        LimitState['涨停溢价率'] = round(self.stockpct[self.Limit_stock.shift(1)==True].mean(axis=1),2).astype(str)+'%'
        LimitState['龙头股溢价率'] = round(self.stockpct[self.Dragon_Stock.shift(1)==True].mean(axis=1),2).astype(str)+'%'
        LimitState['强势股溢价率'] = round(self.stockpct[self.Strong_Stock.shift(1)==True].mean(axis=1),2).astype(str)+'%'

        return LimitState

    # 4、日内情绪：日内涨停溢价率+全A涨幅
    def SentimentInday(self,start_date,end_date):
        # 1、考虑涨停溢价率
        min_close = getData.get_minute_1factor('close', start_datetime=start_date, end_datetime=end_date)
        pre_close = getData.get_daily_1factor('pre_close', date_list=getData.get_date_range(start_date, end_date))
        pre_close = pd.DataFrame(np.array(pre_close.loc[min_close.index.get_level_values('date')]),index=min_close.index, columns=pre_close.columns)
        pct_inday = min_close / pre_close - 1

        Limit_stock_inday = transdaytoinday(self.Limit_stock.shift(1),min_close)

        Limit_Pct = pct_inday[Limit_stock_inday == True].mean(axis=1) #昨日涨停溢价率
        Market_Pct = pct_inday.mean(axis=1)   #全市场A股涨幅

        UseRseult = Limit_Pct+Market_Pct  # Para_Result.mean(axis=1)
        UseRseult = pd.DataFrame(UseRseult, columns=['溢价率'])

        a = UseRseult.reset_index().dropna()
        a['datetime'] = (a['date'] * 10000 + a['time']).astype(str)
        a = a.drop(['date', 'time'], axis=1).set_index('datetime')

        def format_date(x, pos=None):
            if x < 0 or x > len(a.index) - 1:
                return ''
            return a.index[int(x)][:8]

        font = fm.FontProperties(fname='/data/user/015624/STKAITI.TTF')
        fig = plt.subplots(figsize=(30, 10))
        ax1 = plt.subplot(111)
        ax1.plot(a.index, a['溢价率'])
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(242))
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        #for tl in ax1.get_xticklabels():
        #    tl.set_rotation(90)
        ax1.set_title('日内情绪波动（绿色为冰点，红色为高潮）', fontproperties=font, size=24)
        plt.fill_between(a.index, a['溢价率'] - 0.003, a['溢价率'], where=(a['溢价率'] <= -0.005), facecolor='green')
        plt.fill_between(a.index, a['溢价率'] + 0.003, a['溢价率'], where=(a['溢价率'] >= 0.02), facecolor='red')
        # plt.ylim(-0.02, 0.07,0.005)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        plt.grid(axis='y')
        plt.savefig('/data/user/015624/市场情绪/日内情绪波动.png')
        send_file('/data/user/015624/市场情绪/日内情绪波动.png')

    # 5、日间情绪
    def SentimentDaily(self,start_date, end_date):
        Sentiment_Daily = MarketSentiment(start_date, end_date)

        def format_date(x, pos=None):
            if x < 0 or x > len(Sentiment_Daily.index) - 1:
                return ''
            return Sentiment_Daily.index[int(x)]

        font = fm.FontProperties(fname='/data/user/015624/STKAITI.TTF')
        fig = plt.subplots(figsize=(30, 10))
        ax1 = plt.subplot(111)
        ax1.plot(Sentiment_Daily.index, Sentiment_Daily['市场情绪'])
        #ax1.xaxis.set_major_locator(ticker.MultipleLocator(3))
        #ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
        ax1.set_title('情绪走势（0-3:冰点，3-4:过冷，4-6:正常，6-7:过热，7-10:高潮）', fontproperties=font, size=24)
        ax1.yaxis.set_major_locator(MultipleLocator(1))
        plt.grid(axis='y')
        plt.ylim(0,10)
        plt.axhline(y=3, ls="-", c="green")  # 添加水平直线
        plt.axhline(y=7, ls="-", c="red")  # 添加水平直线
        plt.savefig('/data/user/015624/市场情绪/市场情绪走势.png')
        send_file('/data/user/015624/市场情绪/市场情绪走势.png')

    # 6、市场技术层面：市场趋势，市场成交量，
    def Market_Trend(self,start_date,end_date):
        # 市场成交量
        s = FactorData()
        vol = s.get_factor_value('WIND_AIndexEODPrices', factors=['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_AMOUNT'],S_INFO_WINDCODE=['000001.SH', '399001.SZ'], trade_dt=['>=20150101','<='+str(end_date)])
        vol = vol.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_AMOUNT').sum(axis=1)
        BiggerTime = ContinuousTrueTime(vol > 1e9).astype(int)
        Vol_Today = vol.loc[str(start_date):str(end_date)]
        Vol_Today.index = Vol_Today.index.astype(int)
        # 市场趋势
        Trend_Score = MarketIndex_Trend(start_date, end_date)
        Trend_Score['市场成交额（亿)'] = round(Vol_Today/1e5,2)
        send_message('破万亿天数：'+str(BiggerTime.loc[str(end_date)]))

        return Trend_Score

    # 7、赚钱效应
    def MoneyEffect(self,start_date,end_date):
        # 能够代表市场赚钱效应的：成交量位于市场前列，换手率位于市场前列，涨幅不能比较低
        Stock_Buy = ((self.amt.rank(pct=True, axis=1) > 0.9) & (self.turn.rank(pct=True, axis=1) > 0.9) & self.stock_pool)
        # 拆分1-涨停价买入：昨日涨停+昨日炸板
        Limit_buy = (self.Limit_stock | self.OpenBoard_stock) & Stock_Buy
        # 拆分2-追高：昨天未涨停，且最高价＞5%
        High_Buy = ((self.high / self.pre_close - 1) >= 0.05) & Stock_Buy & (Limit_buy == False)
        # 拆分3-抄底买入：最低价＜-5%且最高价低于5%
        Lower_buy = ((self.low / self.pre_close - 1) <= -0.05) & ((self.high / self.pre_close - 1) < 0.05) & Stock_Buy
        # 拆分4：昨天横盘震荡，价格位于-5%-5%之间
        Stay_buy = ((self.low / self.pre_close - 1) > -0.05) & ((self.high / self.pre_close - 1) < 0.05) & Stock_Buy

        s = FactorData()
        IndexResult = s.get_factor_value('WIND_AIndexEODPrices',factors=['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE', 'S_DQ_PCTCHANGE'],
                                         S_INFO_WINDCODE=['000001.SH', '399006.SZ'],trade_dt=['>=' + str(start_date), '<=' + str(end_date)])
        IndexPct = IndexResult.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_PCTCHANGE')
        IndexPct.index = IndexPct.index.astype(int)
        # 计算第二日收益
        Pct = self.vwap / self.pre_close - 1

        MoneyChase = pd.DataFrame(index=get_date_list(start_date,end_date), columns=['指数涨跌幅', '整体赚钱效应', '打板溢价', '追高溢价', '潜伏溢价', '低吸溢价'])
        MoneyChase['整体赚钱效应'] = round(Pct[Stock_Buy.shift(1) == True].mean(axis=1) * 100, 2).astype(str)+'%'
        MoneyChase['打板溢价'] = round(Pct[Limit_buy.shift(1) == True].mean(axis=1) * 100, 2).astype(str)+'%'
        MoneyChase['追高溢价'] = round(Pct[High_Buy.shift(1) == True].mean(axis=1) * 100, 2).astype(str)+'%'
        MoneyChase['潜伏溢价'] = round(Pct[Stay_buy.shift(1) == True].mean(axis=1) * 100, 2).astype(str)+'%'
        MoneyChase['低吸溢价'] = round(Pct[Lower_buy.shift(1) == True].mean(axis=1) * 100, 2).astype(str)+'%'
        MoneyChase['指数涨跌幅'] = round(IndexPct['000001.SH'], 2).astype(str)+'%'

        MoneyChaseNum = pd.DataFrame(index=get_date_list(start_date,end_date), columns=['昨日整体数量', '昨日打板数量', '昨日追高数量', '昨日潜伏数量', '昨日低吸数量'])
        MoneyChaseNum['昨日整体数量'] = Stock_Buy.shift(1).sum(axis=1)
        MoneyChaseNum['昨日打板数量'] = Limit_buy.shift(1).sum(axis=1)
        MoneyChaseNum['昨日追高数量'] = High_Buy.shift(1).sum(axis=1)
        MoneyChaseNum['昨日潜伏数量'] = Stay_buy.shift(1).sum(axis=1)
        MoneyChaseNum['昨日低吸数量'] = Lower_buy.shift(1).sum(axis=1)

        return MoneyChase,MoneyChaseNum

    # 8、输出某日的强势股
    def PowerStock(self,date):
        # 把当日的强势股，龙头股，连板龙头股都输出出来
        stock_name = get_stock_name()
        stock_name.index = pd.Series(stock_name.index).apply(lambda x: stockList.trans_windcode2int(x))
        close_badj = getData.get_daily_1factor('close_badj', date_list=self.date_list)[self.close.columns]
        # 龙头股/妖股
        # 定义1：首先是从文件里读出来的dragon_stock
        # 定义2：近20日涨幅位于市场前100名 & 涨幅＞30% & 距离最高点回撤不能超过50% * 不能是连续跌停（近10日大跌超过-7%以上的日期≥5）
        dragon_stock1 = set(self.Dragon_Stock.shift(1).loc[date][self.Dragon_Stock.shift(1).loc[date] == True].index)
        dragon_stock2 = (self.stockpct.loc[:date].iloc[-20:].sum()>=30) & (self.stockpct.loc[:date].iloc[-20:].sum().rank(ascending=False)<=50) & \
        ((close_badj.loc[date]/close_badj.loc[:date].iloc[-20:].max()-1) / (close_badj.loc[:date].iloc[-20:].max()/ close_badj.loc[date].iloc[-20] - 1)<0.5) & \
            ((self.stockpct.loc[:date].iloc[-10:]<=-7).sum()<5) & self.stock_pool.loc[date]

        dragon_stock2 = set(dragon_stock2[dragon_stock2==True].index)

        dragon_stock = dragon_stock1.union(dragon_stock2)
        Dragon_List = pd.DataFrame(dragon_stock,columns=['龙头股'])

        # 连板股：
        # 定义1：连板龙头股文件里读出来的Limit_Dragon
        # 定义2：二连板及以上的个股
        double_limit_stock1 = set(self.Limit_Dragon.loc[date][self.Limit_Dragon.loc[date] == True].index)
        double_limit_stock2 = set(self.Limit_stock.iloc[:date].iloc[-2:].sum()[self.Limit_stock.iloc[:date].iloc[-2:].sum()>=2].index)

        double_limit_stock = (double_limit_stock1.union(double_limit_stock2)).difference(dragon_stock)
        LimitDragon_List = pd.DataFrame(double_limit_stock, columns = ['连板龙头股'])

        # 强势股：
        # 定义1：从强势股文件里读出来的强势股
        # 定义2：多头排列5>20>60，收盘价＞20，股价在近期高点(即70%分位数之上) & 放量（量比＞1.2）的个股
        kline5 = close_badj.rolling(5).mean()
        kline10 = close_badj.rolling(10).mean()
        kline20 = close_badj.rolling(20).mean()
        kline60 = close_badj.rolling(60).mean()

        strong_stock1 = set(self.Strong_Stock.loc[date][self.Strong_Stock.loc[date] == True].index)

        kline_choice = ((kline5.loc[date]>kline20.loc[date]) & (kline20.loc[date]>kline60.loc[date]) & (close_badj.loc[date]>kline20.loc[date])) | \
                       ((close_badj.loc[date] > kline5.loc[date]) & (kline5.loc[date] > kline10.loc[date]) & (kline10.loc[date] > kline20.loc[date]))


        strong_stock2 =kline_choice & \
        (close_badj.iloc[:date].iloc[-20:].rank(pct=True).loc[date]>0.7) & (self.amt.iloc[:date].iloc[-5:].mean()/self.amt.iloc[:date].iloc[-25:-5].mean()>=1.2)
        strong_stock2 = set(strong_stock2[strong_stock2==True].index)

        strong_stock = (strong_stock1.union(strong_stock2)).difference(double_limit_stock).difference(dragon_stock)
        Strong_List = pd.DataFrame(strong_stock, columns=['强势股'])

        # 修改为名称
        Dragon_List['龙头股'] = Dragon_List['龙头股'].apply(lambda x: stock_name.loc[x])
        LimitDragon_List['连板龙头股'] = LimitDragon_List['连板龙头股'].apply(lambda x: stock_name.loc[x])
        Strong_List['强势股'] = Strong_List['强势股'].apply(lambda x: stock_name.loc[x])

        Stock_List = pd.concat([Dragon_List, LimitDragon_List, Strong_List], axis=1)

        return Stock_List
    ####保存数据#######
    def save_Result(self, delay=5, save_path='/data/user/015624/市场情绪/'):
        Stock_List = self.PowerStock(self.end_date) #获取当日的强势股
        LimitState = self.LimitState_Daily() #日间涨跌停情况
        Trend_Score = self.Market_Trend(self.start_date,self.end_date)  #指数趋势情况
        MoneyChase,MoneyChaseNum = self.MoneyEffect(self.start_date,self.end_date)  #赚钱效应

        self.LimitState_Inday(self.end_date) # 日内涨停情况(图）
        self.SentimentInday(self.date_list[-delay], self.end_date)  # 日内情绪（图）
        self.SentimentDaily(self.start_date, self.end_date) #日间情况（图）
        UpDown_Result, Limit_Result = self.UpDownWeight(self.end_date)  # 日内涨跌比情况（图）

        writer = pd.ExcelWriter(save_path + str(self.end_date) + '历史市场情绪分析.xlsx')
        LimitState.iloc[-delay:].T.to_excel(writer, sheet_name='日内涨跌停情况')
        Trend_Score.iloc[-delay:].to_excel(writer, sheet_name='市场趋势情况')
        MoneyChase.iloc[-delay:].to_excel(writer, sheet_name='赚钱效应情况')
        writer.close()

        Stock_List.to_excel(save_path + str(self.end_date) + '当日强势股.xlsx', sheet_name='当日强势股')

        send_file(save_path + str(self.end_date) + '历史市场情绪分析.xlsx',['015624','011669'])
        send_file(save_path + str(self.end_date) + '当日强势股.xlsx')

start_date,end_date = 20210501,int(datetime.datetime.now().strftime('%Y%m%d')) # 获取今天日期

self =ShortTrade_Sentiment(start_date,end_date)
self.save_Result(delay = 5)