import time
from dataApi.dividend import *
from dataApi.getData import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from SectorRotation.FactorTest import *

def size(start_date,end_date):
    date_list = get_date_range(start_date,end_date)
    mv = get_daily_1factor('mkt_free_cap',date_list=date_list)
    return np.log(mv)
    # type可以选stock或者行业
















