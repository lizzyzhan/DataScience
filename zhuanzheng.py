# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 18:27:00 2017

@author: 10206913
"""

import re
import os
from math import *
import pandas as pd
import numpy as np
import json
import report as rpt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,SVR



from pptx import Presentation





d2,code=rpt.wenjuanwang(filepath='.\\data')


# ================数据清洗=================
d2=d2[d2['Q49'].notnull()]
d2['Q49'].replace({5:u'时尚',6:u'商务',7:u'科技',8:u'跟随'},inplace=True)
code['Q49']['code']={5:u'时尚',6:u'商务',7:u'科技',8:u'跟随'}


'''
code_json=json.dumps(code,ensure_ascii=False)
with open('code.json', 'w') as f:
  f.write(code_json.encode('gbk'))
'''

X=d2[code['Q14']['qlist']]
Y=d2['Q15']
Y.replace({1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:3,11:3},inplace=True)


model1 = LogisticRegression()
model2 = RandomForestRegressor()
model3 = SVC()

model=model1
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print rate1

model=model2
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print rate1

model=model3
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print rate1



  
  
  
