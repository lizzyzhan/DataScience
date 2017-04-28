# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import report as rpt
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Lasso


#  数据d导入
data0=rpt.read_data('data.xlsx')
code=rpt.read_code('code_en.xlsx')

# 数据清洗
data=data0[data0['Q1']==1]

# 数据准备
X=data[code['Q22']['qlist']]
qlist=code['Q22']['qlist']
X=pd.DataFrame(X.as_matrix(),columns=qlist)
Y=data['Q19']
Y.replace({1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5},inplace=True)
X=X.as_matrix()
Y=pd.Series(Y.as_matrix())


model1 = LogisticRegression()
model2 = RandomForestRegressor()
model3 = SVC()
model4 = LinearRegression()
model5 = Lasso(alpha=.3)


model=model4
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)
weight=model.coef_
weight=pd.DataFrame(weight,index=qlist,columns=[u'线性回归'])
tmp=abs(weight)/abs(weight).sum()
tmp.rename(columns={u'线性回归':u'线性回归标准化'},inplace=True)
weight=weight.join(tmp)


model=model2
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
#weight=model.feature_importances_
print(rate1)
print(confusion_matrix)
tmp=model.feature_importances_
tmp=pd.DataFrame(tmp,index=qlist,columns=[u'随机森林'])
weight=weight.join(tmp)




'''
model=model3
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)

model=model1
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)
'''

tmp1=pd.DataFrame(np.mean(X,axis=0),index=qlist,columns=[u'满意度'])
weight=weight.join(tmp1)
tmp2=tmp1/tmp1.sum().sum()
tmp2.rename(columns={u'满意度':u'满意度标准化'},inplace=True)
weight=weight.join(tmp2)

Y_corr=pd.Series([np.corrcoef(X[:,qq],Y)[0,1] for qq in range(len(qlist))],index=qlist)
weight=weight.join(pd.DataFrame(Y_corr,columns=[u'相关系数']))

# 相关系数


'''
for qq in code['Q14']['qlist']:
    print(code['Q14']['code_r'][qq])
    plt.figure(1)
    plt.scatter(d2[qq],d2['Q15'])
    plt.show()
'''

#from scipy.stats import kendalltau
#sns.jointplot(d2[qq],d2['Q15'],kind="hex",stat_func=kendalltau,color="#4CB391")

'''
# 气泡图
for qq in code['Q21']['qlist']:
    print(code['Q21']['code_r'][qq])
    plt.figure(1)
    x=[]
    y=[]
    z=[]
    k=0
    t=pd.crosstab(data[qq],data['Q18']).stack()
    x=[w[0] for w in t.index]
    y=[w[1] for w in t.index]
    z=list(t)
    tt=pd.DataFrame({'x':x,'y':y,'z':z})
    #tt=pd.DataFrame({'x':x,'y':y,'z':z})
    plt.scatter(tt['x'],tt['y'],tt['z']*30,color='orange',alpha=0.6)
    tt.to_csv(qq+'.csv',index=False)
    plt.show()
'''    
