# -*- coding: utf-8 -*-
import http.client
import traceback
import pandas as pd
import urllib
import gzip
from io import BytesIO

HTTP_OK = 200
HTTP_AUTHORIZATION_ERROR = 401
class Client:
    domain = 'api.wmcloud.com'
    port = 443
    token = '9597973eb7038839a12e4763075c70d62c6c254652dddb512ea3f08feda4b14d'
    #设置因网络连接，重连的次数
    reconnectTimes=2
    httpClient = None
    def __init__( self ):
        self.httpClient = http.client.HTTPSConnection(self.domain, self.port, timeout=60)
    def __del__( self ):
        if self.httpClient is not None:
            self.httpClient.close()
    def encodepath(self, path):
        #转换参数的编码
        start=0
        n=len(path)
        re=''
        i=path.find('=',start)
        while i!=-1 :
            re+=path[start:i+1]
            start=i+1
            i=path.find('&',start)
            if(i>=0):
                for j in range(start,i):
                    if(path[j]>'~'):
                        re+=urllib.parse.quote(path[j])
                    else:
                        re+=path[j]
                re+='&'
                start=i+1
            else:
                for j in range(start,n):
                    if(path[j]>'~'):
                        re+=urllib.parse.quote(path[j])
                    else:
                        re+=path[j]
                start=n
            i=path.find('=',start)
        return re
    def init(self, token):
        self.token=token
    def getData(self, path):
        result = None
        path='/data/v1' + path
        print (path)
        path=self.encodepath(path)
        for i in range(self.reconnectTimes):
            try:
                #set http header here
                self.httpClient.request('GET', path, headers = {"Authorization": "Bearer " + self.token,
                                                                "Accept-Encoding": "gzip, deflate"})
                #make request
                response = self.httpClient.getresponse()
                result = response.read()
                compressedstream = BytesIO(result)
                gziper = gzip.GzipFile(fileobj=compressedstream)
                try:
                    result = gziper.read()
                except:
                    pass
                return response.status, result
            except Exception as e:
                if i == self.reconnectTimes-1:
                    raise e
                if self.httpClient is not None:
                    self.httpClient.close()
                self.httpClient = http.client.HTTPSConnection(self.domain, self.port, timeout=60)
        return -1, result

self = Client()
self.init('9597973eb7038839a12e4763075c70d62c6c254652dddb512ea3f08feda4b14d')
URL = '/api/HKequity/getEquShszStats.json?field=&beginDate=20170101&endDate=20190101&shcID=&shcName=&partyID='
code,result = self.getData(path = URL)
if code == 200:
    print(result.decode('utf-8',errors='replace'))
    if eval(result)['retCode']==1:
        pd_data = pd.DataFrame(eval(result)['data'])

data_result = pd_data.pivot_table(index='tradeDate',columns='shcName',values='netFlow')
data_result.index = pd.Series(data_result.index).apply(lambda x:int(x.replace('-','')))

data_result1 = pd_data.pivot_table(index='tradeDate',columns='shcName',values='netFlow')
data_result1.index = pd.Series(data_result1.index).apply(lambda x:int(x.replace('-','')))

data_result2 = pd_data.pivot_table(index='tradeDate',columns='shcName',values='netFlow')
data_result2.index = pd.Series(data_result2.index).apply(lambda x:int(x.replace('-','')))

data_result.columns = pd.Series(data_result.columns).apply(lambda x: '银行' if (('银行' in x) or('BANK' in x)) else '证券')




all_data = pd.read_pickle('E:/ABasicData/north_compnay.pkl')
all_data.columns = pd.Series(all_data.columns).apply(lambda x: '银行' if (('银行' in x) or('BANK' in x)) else '证券')

north_company = pd.concat([all_data['证券'].sum(axis=1).rename('证券'),all_data['银行'].sum(axis=1).rename('银行')],axis=1)
north_company





all_data = pd.concat([data_result2,data_result1,data_result]).sort_index()
all_data.to_pickle('E:/ABasicData/north_compnay.pkl')
