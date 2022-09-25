import requests, sys, json
import pandas as pd
import datetime, os
import numpy as np
from ftplib import FTP
from dataApi import getData, stockList, tradeDate


# 函数1：获取周期内前推N日的日期
def get_date_list(start_date, end_date, delay=0):
    if delay == 'history':
        return getData.get_date_range(int(20100101), int(end_date))
    else:
        new_start_date = tradeDate.get_pre_trade_date(start_date,offset=delay)
        date_list = getData.get_date_range(int(start_date), int(end_date))
        start_date = date_list[0]
        date_list_before = getData.get_date_range(20100101, int(end_date))
        return date_list_before[max(date_list_before.index(start_date) - delay, 0):]







# 函数5：获取申万行业及成分股（一级，二级）
def SWIndustry_Pool(now_date, level='SW1', way='dict'):
    if level == 'SW1':
        code_dict = {
            '801020.SI': '采掘（SW）',
            '801040.SI': '钢铁（SW）',
            '801050.SI': '有色金属（SW）',
            '801030.SI': '化工（SW）',
            '801710.SI': '建筑材料（SW）',
            '801170.SI': '交通运输（SW）',
            '801740.SI': '国防军工（SW）',
            '801890.SI': '机械设备（SW）',
            '801730.SI': '电气设备（SW）',
            '801720.SI': '建筑装饰（SW）',
            '801010.SI': '农林牧渔（SW）',
            '801140.SI': '轻工制造（SW）',
            '801120.SI': '食品饮料（SW）',
            '801130.SI': '纺织服装（SW）',
            '801110.SI': '家用电器（SW）',
            '801880.SI': '汽车（SW）',
            '801200.SI': '商业贸易（SW）',
            '801210.SI': '休闲服务（SW）',
            '801750.SI': '计算机（SW）',
            '801080.SI': '电子（SW）',
            '801770.SI': '通信（SW）',
            '801760.SI': '传媒（SW）',
            '801780.SI': '银行（SW）',
            '801790.SI': '非银金融（SW）',
            '801180.SI': '房地产（SW）',
            '801150.SI': '医药生物（SW）',
            '801160.SI': '公用事业（SW）',
            '801230.SI': '综合（SW）',
        }
    elif level == 'SW2':
        code_dict = {
            '801024.SI': '采掘服务',
            '801021.SI': '煤炭开采Ⅱ',
            '801022.SI': '其他采掘Ⅱ',
            '801023.SI': '石油开采Ⅱ',
            '801041.SI': '钢铁Ⅱ',
            '801055.SI': '工业金属',
            '801053.SI': '黄金Ⅱ',
            '801051.SI': '金属非金属新材料',
            '801054.SI': '稀有金属',
            '801032.SI': '化学纤维',
            '801033.SI': '化学原料',
            '801034.SI': '化学制品',
            '801035.SI': '石油化工',
            '801036.SI': '塑料',
            '801037.SI': '橡胶',
            '801712.SI': '玻璃制造Ⅱ',
            '801713.SI': '其他建材Ⅱ',
            '801711.SI': '水泥制造Ⅱ',
            '801171.SI': '港口Ⅱ',
            '801175.SI': '高速公路Ⅱ',
            '801172.SI': '公交Ⅱ',
            '801173.SI': '航空运输Ⅱ',
            '801176.SI': '航运Ⅱ',
            '801174.SI': '机场Ⅱ',
            '801177.SI': '铁路运输Ⅱ',
            '801178.SI': '物流Ⅱ',
            '801744.SI': '船舶制造Ⅱ',
            '801743.SI': '地面兵装Ⅱ',
            '801742.SI': '航空装备Ⅱ',
            '801741.SI': '航天装备Ⅱ',
            '801075.SI': '金属制品Ⅱ',
            '801072.SI': '通用机械',
            '801073.SI': '仪器仪表Ⅱ',
            '801076.SI': '运输设备Ⅱ',
            '801074.SI': '专用设备',
            '801731.SI': '电机Ⅱ',
            '801732.SI': '电气自动化设备',
            '801733.SI': '电源设备',
            '801734.SI': '高低压设备',
            '801721.SI': '房屋建设Ⅱ',
            '801723.SI': '基础建设',
            '801725.SI': '园林工程Ⅱ',
            '801724.SI': '专业工程',
            '801722.SI': '装修装饰Ⅱ',
            '801017.SI': '畜禽养殖Ⅱ',
            '801018.SI': '动物保健Ⅱ',
            '801011.SI': '林业Ⅱ',
            '801012.SI': '农产品加工',
            '801013.SI': '农业综合Ⅱ',
            '801014.SI': '饲料Ⅱ',
            '801015.SI': '渔业',
            '801016.SI': '种植业',
            '801141.SI': '包装印刷Ⅱ',
            '801142.SI': '家用轻工',
            '801144.SI': '其他轻工制造Ⅱ',
            '801143.SI': '造纸Ⅱ',
            '801124.SI': '食品加工',
            '801123.SI': '饮料制造',
            '801131.SI': '纺织制造',
            '801132.SI': '服装家纺',
            '801111.SI': '白色家电',
            '801112.SI': '视听器材',
            '801881.SI': '其他交运设备Ⅱ',
            '801092.SI': '汽车服务Ⅱ',
            '801093.SI': '汽车零部件Ⅱ',
            '801094.SI': '汽车整车',
            '801202.SI': '贸易Ⅱ',
            '801205.SI': '商业物业经营',
            '801203.SI': '一般零售',
            '801204.SI': '专业零售',
            '801211.SI': '餐饮Ⅱ',
            '801212.SI': '景点',
            '801213.SI': '酒店Ⅱ',
            '801214.SI': '旅游综合Ⅱ',
            '801215.SI': '其他休闲服务Ⅱ',
            '801101.SI': '计算机设备Ⅱ',
            '801222.SI': '计算机应用',
            '801081.SI': '半导体',
            '801085.SI': '电子制造',
            '801084.SI': '光学光电子',
            '801082.SI': '其他电子Ⅱ',
            '801083.SI': '元件Ⅱ',
            '801102.SI': '通信设备',
            '801223.SI': '通信运营Ⅱ',
            '801752.SI': '互联网传媒',
            '801761.SI': '文化传媒',
            '801751.SI': '营销传播',
            '801192.SI': '银行Ⅱ',
            '801194.SI': '保险Ⅱ',
            '801191.SI': '多元金融Ⅱ',
            '801193.SI': '证券Ⅱ',
            '801181.SI': '房地产开发Ⅱ',
            '801182.SI': '园区开发Ⅱ',
            '801151.SI': '化学制药',
            '801152.SI': '生物制品Ⅱ',
            '801156.SI': '医疗服务Ⅱ',
            '801153.SI': '医疗器械Ⅱ',
            '801154.SI': '医药商业Ⅱ',
            '801155.SI': '中药Ⅱ',
            '801161.SI': '电力',
            '801162.SI': '环保工程及服务Ⅱ',
            '801163.SI': '燃气Ⅱ',
            '801164.SI': '水务Ⅱ',
            '801231.SI': '综合Ⅱ',
        }
    code = list(code_dict.keys())
    name = list(code_dict.values())
    s = FactorData()
    SWStock_list = s.get_factor_value('WIND_SWIndexMembers',
                                      factors=['S_INFO_WINDCODE', 'S_CON_WINDCODE', 'S_CON_INDATE', 'S_CON_OUTDATE'],
                                      S_INFO_WINDCODE=code, S_CON_INDATE=['<=' + str(now_date)])
    SWStock_list = SWStock_list[SWStock_list['S_CON_OUTDATE'].isna() | (SWStock_list['S_CON_OUTDATE'] > str(now_date))]
    SWStock_list['S_INFO_WINDCODE'] = SWStock_list['S_INFO_WINDCODE'].apply(lambda x: code_dict[x])
    SWStock_list['S_CON_INDATE'] = True
    if way == 'df':
        IndustryList = SWStock_list.pivot_table(index='S_INFO_WINDCODE', columns='S_CON_WINDCODE',
                                                values='S_CON_INDATE').fillna(False)
        IndustryList.columns = pd.Series(IndustryList.columns).apply(lambda x: getData.trans_windcode2int(x))
    elif way == 'dict':
        IndustryList = {}
        for SWcode in name:
            IndustryList[SWcode] = set([stockList.trans_windcode2int(x) for x in
                                        set(SWStock_list[SWStock_list['S_INFO_WINDCODE'] == SWcode]['S_CON_WINDCODE'])])

    return IndustryList, code_dict


# （1）获取申万行业因子（一级，二级）
def SWIndustry_Facotr(factor, start_date, end_date, index='SW1'):
    factor_list = {'pct': 'pct', 'close': 'S_DQ_CLOSE', 'amt': 'S_DQ_AMOUNT', 'volume': 'S_DQ_VOLUME', 'mv': 'S_VAL_MV',
                   'free_mv': 'S_DQ_MV', 'pre_close': 'S_DQ_PRECLOSE'}
    factor = factor_list[factor]
    if index == 'SW1':
        code_dict = {
            '801020.SI': '采掘（SW）',
            '801040.SI': '钢铁（SW）',
            '801050.SI': '有色金属（SW）',
            '801030.SI': '化工（SW）',
            '801710.SI': '建筑材料（SW）',
            '801170.SI': '交通运输（SW）',
            '801740.SI': '国防军工（SW）',
            '801890.SI': '机械设备（SW）',
            '801730.SI': '电气设备（SW）',
            '801720.SI': '建筑装饰（SW）',
            '801010.SI': '农林牧渔（SW）',
            '801140.SI': '轻工制造（SW）',
            '801120.SI': '食品饮料（SW）',
            '801130.SI': '纺织服装（SW）',
            '801110.SI': '家用电器（SW）',
            '801880.SI': '汽车（SW）',
            '801200.SI': '商业贸易（SW）',
            '801210.SI': '休闲服务（SW）',
            '801750.SI': '计算机（SW）',
            '801080.SI': '电子（SW）',
            '801770.SI': '通信（SW）',
            '801760.SI': '传媒（SW）',
            '801780.SI': '银行（SW）',
            '801790.SI': '非银金融（SW）',
            '801180.SI': '房地产（SW）',
            '801150.SI': '医药生物（SW）',
            '801160.SI': '公用事业（SW）',
            '801230.SI': '综合（SW）',
        }
    elif index == 'SW2':
        code_dict = {
            '801024.SI': '采掘服务',
            '801021.SI': '煤炭开采Ⅱ',
            '801022.SI': '其他采掘Ⅱ',
            '801023.SI': '石油开采Ⅱ',
            '801041.SI': '钢铁Ⅱ',
            '801055.SI': '工业金属',
            '801053.SI': '黄金Ⅱ',
            '801051.SI': '金属非金属新材料',
            '801054.SI': '稀有金属',
            '801032.SI': '化学纤维',
            '801033.SI': '化学原料',
            '801034.SI': '化学制品',
            '801035.SI': '石油化工',
            '801036.SI': '塑料',
            '801037.SI': '橡胶',
            '801712.SI': '玻璃制造Ⅱ',
            '801713.SI': '其他建材Ⅱ',
            '801711.SI': '水泥制造Ⅱ',
            '801171.SI': '港口Ⅱ',
            '801175.SI': '高速公路Ⅱ',
            '801172.SI': '公交Ⅱ',
            '801173.SI': '航空运输Ⅱ',
            '801176.SI': '航运Ⅱ',
            '801174.SI': '机场Ⅱ',
            '801177.SI': '铁路运输Ⅱ',
            '801178.SI': '物流Ⅱ',
            '801744.SI': '船舶制造Ⅱ',
            '801743.SI': '地面兵装Ⅱ',
            '801742.SI': '航空装备Ⅱ',
            '801741.SI': '航天装备Ⅱ',
            '801075.SI': '金属制品Ⅱ',
            '801072.SI': '通用机械',
            '801073.SI': '仪器仪表Ⅱ',
            '801076.SI': '运输设备Ⅱ',
            '801074.SI': '专用设备',
            '801731.SI': '电机Ⅱ',
            '801732.SI': '电气自动化设备',
            '801733.SI': '电源设备',
            '801734.SI': '高低压设备',
            '801721.SI': '房屋建设Ⅱ',
            '801723.SI': '基础建设',
            '801725.SI': '园林工程Ⅱ',
            '801724.SI': '专业工程',
            '801722.SI': '装修装饰Ⅱ',
            '801017.SI': '畜禽养殖Ⅱ',
            '801018.SI': '动物保健Ⅱ',
            '801011.SI': '林业Ⅱ',
            '801012.SI': '农产品加工',
            '801013.SI': '农业综合Ⅱ',
            '801014.SI': '饲料Ⅱ',
            '801015.SI': '渔业',
            '801016.SI': '种植业',
            '801141.SI': '包装印刷Ⅱ',
            '801142.SI': '家用轻工',
            '801144.SI': '其他轻工制造Ⅱ',
            '801143.SI': '造纸Ⅱ',
            '801124.SI': '食品加工',
            '801123.SI': '饮料制造',
            '801131.SI': '纺织制造',
            '801132.SI': '服装家纺',
            '801111.SI': '白色家电',
            '801112.SI': '视听器材',
            '801881.SI': '其他交运设备Ⅱ',
            '801092.SI': '汽车服务Ⅱ',
            '801093.SI': '汽车零部件Ⅱ',
            '801094.SI': '汽车整车',
            '801202.SI': '贸易Ⅱ',
            '801205.SI': '商业物业经营',
            '801203.SI': '一般零售',
            '801204.SI': '专业零售',
            '801211.SI': '餐饮Ⅱ',
            '801212.SI': '景点',
            '801213.SI': '酒店Ⅱ',
            '801214.SI': '旅游综合Ⅱ',
            '801215.SI': '其他休闲服务Ⅱ',
            '801101.SI': '计算机设备Ⅱ',
            '801222.SI': '计算机应用',
            '801081.SI': '半导体',
            '801085.SI': '电子制造',
            '801084.SI': '光学光电子',
            '801082.SI': '其他电子Ⅱ',
            '801083.SI': '元件Ⅱ',
            '801102.SI': '通信设备',
            '801223.SI': '通信运营Ⅱ',
            '801752.SI': '互联网传媒',
            '801761.SI': '文化传媒',
            '801751.SI': '营销传播',
            '801192.SI': '银行Ⅱ',
            '801194.SI': '保险Ⅱ',
            '801191.SI': '多元金融Ⅱ',
            '801193.SI': '证券Ⅱ',
            '801181.SI': '房地产开发Ⅱ',
            '801182.SI': '园区开发Ⅱ',
            '801151.SI': '化学制药',
            '801152.SI': '生物制品Ⅱ',
            '801156.SI': '医疗服务Ⅱ',
            '801153.SI': '医疗器械Ⅱ',
            '801154.SI': '医药商业Ⅱ',
            '801155.SI': '中药Ⅱ',
            '801161.SI': '电力',
            '801162.SI': '环保工程及服务Ⅱ',
            '801163.SI': '燃气Ⅱ',
            '801164.SI': '水务Ⅱ',
            '801231.SI': '综合Ⅱ',
        }
    SW_list = list(code_dict.keys())
    s = FactorData()
    if factor == 'pct':
        SW_Close = s.get_factor_value('WIND_ASWSIndexEOD', factors=['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE'],
                                      S_INFO_WINDCODE=SW_list, trade_dt=['>=' + str(start_date), '<=' + str(end_date)])
        concept_close = SW_Close.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='S_DQ_CLOSE')
        concept_close.index = concept_close.index.astype(int)
        concept_pct = concept_close.pct_change(1) * 100
        concept_pct.columns = pd.Series(concept_pct.columns).apply(lambda x: code_dict[x])
        return concept_pct

    else:
        SW_factor = s.get_factor_value('WIND_ASWSIndexEOD', factors=['S_INFO_WINDCODE', 'TRADE_DT', factor],
                                       S_INFO_WINDCODE=SW_list, trade_dt=['>=' + str(start_date), '<=' + str(end_date)])
        concept_factor = SW_factor.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values=factor)
        concept_factor.index = concept_factor.index.astype(int)
        concept_factor.columns = pd.Series(concept_factor.columns).apply(lambda x: code_dict[x])
        if (factor == 'S_VAL_MV') or (factor == 'S_DQ_MV'):
            concept_factor = concept_factor * 10000
        return concept_factor


# （3）获取市场板块的情况
def market_concept_factor(factor, start_date, end_date, type='All', date=20211025, Together=True):
    if type == 'THS':
        result = THSConcept_Factor(factor, start_date, end_date, date=date, Together=Together)
    if (type == 'SW1') or ((type == 'SW2')):
        result = SWIndustry_Facotr(factor, start_date, end_date, index=type)
    if type == 'All':
        THS_result = THSConcept_Factor(factor, start_date, end_date, date=date, Together=Together)
        SW_result = SWIndustry_Facotr(factor, start_date, end_date, index='SW1')
        result = pd.concat([THS_result, SW_result], axis=1)

    return result


# 函数7：计算涨跌停价格
def cal_limit_price(pre_close):
    Limit_price = round(pre_close * 1.1 + 0.0001, 2)
    stock_pool_688 = pd.Series(Limit_price.columns).apply(
        lambda x: x if (x // 1000 == 688) else np.nan).dropna().astype(int)
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

    return Limit_price, Lowest_Price


# 获取股票名称
def get_stock_name():
    s = FactorData()
    Stock_Name = s.get_factor_value('Basic_factor', [], [datetime.datetime.now().strftime('%Y%m%d')], ['short_name'])[
        'short_name']

    return Stock_Name


