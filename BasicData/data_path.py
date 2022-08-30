import os,datetime
import pickle
from dataApi.TradeDate import _check_input_date
from dataApi.StockList import *
import pandas as pd
import numpy as np
import cx_Oracle
import gc,re

base_address = 'D:/Program/BasicData/'
stock_address = base_address + 'stock/'
bench_address = base_address + 'bench/'


