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



import pandas as pd
import amazon




pid_info=pd.read_csv('phone_pid_de_info.csv',index_col='pid')
pid_info['pid']=list(pid_info.index)
pid_list=list(pid_info[pid_info['country'].notnull()].index)


i=0
for pid in pid_list:
    i+=1
    info=amazon.get_info(pid)
    pid_info.loc[pid,:]=pd.Series(info)
    print('get %s (%d/%d)'%(pid,i,len(pid_list)))
   
pid_info.to_csv('phone_pid_de_info.csv',index=False)

