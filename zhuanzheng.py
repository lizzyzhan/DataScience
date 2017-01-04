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
import matplotlib.pyplot as plt
import seaborn as sns
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
#Y.replace({1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:3,11:3},inplace=True)
#Y.replace({1:1,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:4,10:5,11:5},inplace=True)
Y=pd.Series(Y.as_matrix())

model1 = LogisticRegression()
model2 = RandomForestRegressor()
model3 = SVC()

model=model1
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)

model=model2
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)

model=model3
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)




# 相关系数

for qq in code['Q14']['qlist']:
    r=np.corrcoef(d2[qq],d2['Q15'])[0,1]
    print(code['Q14']['code_r'][qq]+',%.2f'%r)


for qq in code['Q14']['qlist']:
    print(code['Q14']['code_r'][qq])
    plt.figure(1)
    plt.scatter(d2[qq],d2['Q15'])
    plt.show()

from scipy.stats import kendalltau
sns.jointplot(d2[qq],d2['Q15'],kind="hex",stat_func=kendalltau,color="#4CB391")

# 气泡图
for qq in code['Q14']['qlist']:
    print(code['Q14']['code_r'][qq])
    plt.figure()
    x=[]
    y=[]
    z=[]
    k=0
    t=pd.crosstab(d2[qq],d2['Q15']).stack()
    for i in range(5):
        for j in range(11):
            x.append(i+1)
            y.append(j+1)
            z.append(t[i+1][j+1])
            k=k+1
    tt=pd.DataFrame({'x':x,'y':y,'z':z})
    plt.scatter(tt['x'],tt['y'],tt['z']*30,alpha=0.6)
    plt.show()






