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
URL = '/api/HKequity/getEquShszStats.json?field=&beginDate=20220801&endDate=20220906&shcID=&shcName=&partyID='
code,result = self.getData(path = URL)
if code == 200:
    print(result.decode('utf-8',errors='replace'))
    if eval(result)['retCode']==1:
        pd_data = pd.DataFrame(eval(result)['data'])

data_result = pd_data.pivot_table(index='tradeDate',columns='shcName',values='netFlow')
data_result.index = pd.Series(data_result.index).apply(lambda x:int(x.replace('-','')))

a= pd.Series(data_result.columns).apply(lambda x:'证券' if (('证券' in x) or('SECURITIES' in x)) else
   '银行' if (('银行' in x) or('BANK' in x)) else x)


a.sort_values()


x = '2022-08-01'


from io import StringIO

pd.read_csv(StringIO(result[1].decode()),encoding='utf_8_sig')
StringIO(result[1].decode()).to_csv(encoding='utf_8_sig')

len(result[1])

import httpx

headers = {'header': 'Authorization: Bearer < ac3bed137eaf1c6332c3c8c27f365652cfe77fb4d2804f3dfd5489e1eea9985b >'}
params = {'protocol': 'https', 'method': 'GET'}
url = 'https://api.wmcloud.com/data/v1/api/HKequity/getHKshszHold.json?field=&beginDate=20171101&endDate=20171101'
r = httpx.get(url,headers=headers,params=params)
r.text





