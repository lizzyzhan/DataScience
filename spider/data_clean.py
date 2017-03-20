# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:16:35 2017

@author: JSong
"""

# 数据预处理

import os
import pandas as pd



filelist=os.listdir('.\\data\\')
pid=[os.path.splitext(f)[0] for f in filelist]

data=pd.DataFrame(columns=['pid','country','date','author','title','rating','review'])

i=0
for f in filelist:
    print(i)
    d=pd.read_csv('.\\data\\'+f,encoding='ANSI')
    i+=1
    if d.empty:
        continue
    data=data.append(d,ignore_index=True)
data=pd.DataFrame(data,columns=['pid','country','date','author','title','rating','review'])
data.to_csv('reviews_amazon_de.csv',index=False,encoding='utf-8')




import pandas as pd
from textblob import TextBlob

data=pd.read_csv('reviews_amazon_de.csv')
s=TextBlob(data.iloc[1,-1])


