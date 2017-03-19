# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:38:01 2017

@author: JSong
"""

from urllib import request
from bs4 import BeautifulSoup
import pandas as pd
import re




def get_data(url):
    # 2177:年龄中位数
    # 2010：年龄结构
    #url='https://www.cia.gov/library/publications/the-world-factbook/fields/2177.html'
    req = request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1')
    data=request.urlopen(req,timeout=200).read()
    html=BeautifulSoup(data.decode('utf-8','ignore'),'lxml')
    result={}
    h=html.findAll('tr')
    for ht in h[1:]:
        c=ht.findAll('td',{'class':'country'})
        country=c[0].text
        result[country]={}
        name=re.findall('<strong>(.*?)</strong>',str(ht))
        value=re.findall('</strong>(.*?)<br/>',str(ht))
        result[country]=dict(zip(name,value))
    
    result=pd.DataFrame(result).T
    result=result.reset_index()
    result.rename(columns={'index':'country'},inplace=True)
    return result


'''
columns=result.columns[1:]
for c in columns:
    tmp=result[c]
    result[c]=tmp.map(lambda x: re.findall('^[\d\.]{1,}',x)[0])
result.to_csv('median_age.csv',index=False)
'''




    
