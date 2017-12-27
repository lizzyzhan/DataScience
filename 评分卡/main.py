# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:14:04 2017

@author: gason
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import reportgen as rpt




datadict=pd.read_excel('.\\LendingClubData\\LCDataDictionary.xlsx')

columnnew=dict(zip(datadict.loc[datadict['中文名称'].notnull(),'LoanStatNew'],datadict.loc[datadict['中文名称'].notnull(),'中文名称']))
data=pd.read_csv('.\\LendingClubData\\LoanStats_2016Q4.csv',skiprows=1)

data=data.rename(columns=columnnew)
missing_pct=data.apply(lambda x : (len(x)-x.count())/len(x))
#rpt.AnalysisReport(data,'LendingClub数据概览');


# 去除缺失率大于80%的字段
missing_pct=data.apply(lambda x : (len(x)-x.count())/len(x))
data=data.loc[:,missing_pct[missing_pct<0.80].index]

# 去除缺失率大于80%的字段
data=data.dropna(thresh=len(data)*0.5,axis=1)

# 去除那些只有一个类别的字段
data=data.loc[:,data.apply(pd.Series.nunique)!=1]

VarType=rpt.analysis.type_of_var(data)
VarType=pd.Series(VarType)

continuous_var=list(VarType[VarType=='number'].index)
print('数值变量有：\n',' , '.join(continuous_var))

categorical_var=list(VarType[VarType=='category'].index)
print('因子变量有：\n',' , '.join(categorical_var))

datetime_var=list(VarType[VarType=='datetime'].index)
print('时间变量有：\n',' , '.join(datetime_var))

text_var=list(VarType[VarType=='text'].index)
print('文本变量有：\n',' , '.join(text_var))


# 处理

# 处理目标变量
data['target']=data['target'].replace({'Current':np.nan,'Fully Paid':0,'Charged Off':1,'Late (31-120 days)':1,'In Grace Period':np.nan,\
                                      'Late (16-30 days)':1,'Default':np.nan})
data=data.loc[data['target'].notnull(),:]

# 特征衍生
data['放款日期']=pd.to_datetime(data['放款日期'])
data['信用报告最早日期']=pd.to_datetime(data['信用报告最早日期'])
data['days']=(data['放款日期']-data['信用报告最早日期']).map(lambda x:x.days)
data=data.drop(['放款日期','信用报告最早日期'],axis=1)

categorical_var.remove('target')
continuous_var.append('days')

from sklearn.preprocessing import Imputer


# 补缺
# 对于连续变量，这里我们随意填入
for v in continuous_var:
    s=list(data[v].dropna().unique())
    data[v]=data[v].map(lambda x : np.random.choice(s,1)[0] if '%s'%x=='nan' else x)
# 也可以直接用
data.loc[:,continuous_var]=Imputer(strategy='mean').fit_transform(data.loc[:,continuous_var])
# 对于因子变量，这里我们填入众数
for v in categorical_var:
    s=data[v].value_counts().argmax()
    data[v]=data[v].map(lambda x : s if '%s'%x=='nan' else x)

data=data.reset_index(drop=True)#便于后续数据处理，不解释
data0=data.copy()#备份用

# 补缺后，我们可以特征编码啦
# 这里有几种方式，我们待会会一个一个试
# 1、因子变量哑变量,数值变量标准化
# 2、因子变量WOE，数值变量标准化
# 3、数值变量离散后再WOE，同时因子变量WOE


from metrics import WeightOfEvidence
from discretization import Discretization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
max_intervals=8

N=len(data)


'''=========方法一=========='''

#标准化，返回值为标准化后的数据
data_stand=StandardScaler().fit_transform(data.loc[:,continuous_var])
#哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
data_dummies=pd.get_dummies(data.loc[:,categorical_var], prefix=categorical_var,drop_first=True)
data=pd.concat([data_stand,data_dummies],axis=1)



'''=========方法二=========='''
#标准化，返回值为标准化后的数据
data.loc[:,continuous_var]=StandardScaler().fit_transform(data.loc[:,continuous_var])
woe_iv={}# 用于之后的特征选择
for v in categorical_var:
    # 如果因子数过多，则先进行分箱
    if not(isinstance(data[v].iloc[0],str)) and len(data[v].unique())>20:       
        dis=Discretization(method='chimerge',max_intervals=max_intervals)
        dis.fit(data[v],data['target'])
        data.loc[:,v]=dis.transform(data[v])
    woe=WeightOfEvidence()
    woe.fit(data[v],data['target'])
    woe_iv[v]={'woe':woe.woe,'iv':woe.iv}
    data[v]=data[v].replace(woe.woe).astype(np.float64)


'''=========方法三=========='''
woe_iv={}# 用于之后的特征选择
for v in categorical_var:
    # 如果因子数过多，则先进行分箱
    if not(isinstance(data[v].iloc[0],str)) and len(data[v].unique())>20:       
        dis=Discretization(method='chimerge',max_intervals=max_intervals)
        dis.fit(data[v],data['target'])
        data.loc[:,v]=dis.transform(data[v])
    woe=WeightOfEvidence()
    woe.fit(data[v],data['target'])
    woe_iv[v]={'woe':woe.woe,'iv':woe.iv}
    data[v]=data[v].replace(woe.woe).astype(np.float64)

for v in continuous_var:
    print(v)
    if len(data[v].unique())>max_intervals:
        dis=Discretization(method='chimerge',max_intervals=max_intervals,sample=1000)
        dis.fit(data[v],data['target'])
        data.loc[:,v]=dis.transform(data[v])
    woe=WeightOfEvidence()
    woe.fit(data[v],data['target'])
    woe_iv[v]={'woe':woe.woe,'iv':woe.iv}
    data[v]=data[v].replace(woe.woe).astype(np.float64)


# 按照卡方统计、方差分析、信息量来筛选特征。
# 


'''====================特征筛选=========================='''


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#选择K个最好的特征，返回选择特征后的数据
SelectKBest(chi2, k=2).fit_transform(data, data['target'])


iv_threshould=0.01
varByIV=[k for k,v in woe_iv.items() if v['iv'] > iv_threshould]
print('剩余%d个变量'%len(varByIV))
print(varByIV)

# 此处还可以进行多变量分析，共线性分析等方法来去除变量
var_WOE_list=varByIV
X=data[var_WOE_list]
y=data['target']



'''
# 不平衡问题重采样
print('采样前')
n_sample = y.shape[0]
n_pos_sample = y[y==0].shape[0]
n_neg_sample = y[y==1].shape[0]
print('特征个数：',X.shape[1])
print()
print('样本个数{}，正常样本{},逾期样本{},逾期样本占{:.0%}'.format(n_sample,n_pos_sample,n_neg_sample,n_neg_sample/n_sample))

from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=1)
X,y=smote.fit_sample(X,y)


print('采样后')
n_sample = y.shape[0]
n_pos_sample = y[y==0].shape[0]
n_neg_sample = y[y==1].shape[0]
print('特征个数：',X.shape[1])
print()
print('样本个数{}，正常样本{},逾期样本{},逾期样本占{:.0%}'.format(n_sample,n_pos_sample,n_neg_sample,n_neg_sample/n_sample))
'''


'''逻辑回归'''
from sklearn.cross_validation import train_test_split as sp
from sklearn.linear_model import LogisticRegression as LR

X_train,X_test,y_train,y_test=sp(X,y,test_size=0.3)

lr=LR()
lr.fit(X_train,y_train)
y_train_label=lr.predict(X_train)
y_test_label=lr.predict(X_test)

from sklearn.metrics import accuracy_score
print('训练集准确率：{:.2%}'.format(accuracy_score(y_train_label,y_train)))
print('测试集准确率：{:.2%}'.format(accuracy_score(y_test_label,y_test)))



from sklearn.cross_validation import train_test_split as sp
X_train,X_test,y_train,y_test=sp(X,y,test_size=0.3,random_stated=1)




col=['loan_amnt','term','int_rate','installment','grade','sub_grade','emp_length','home_ownership','annual_inc','verification_status',\
     'issue_d','loan_status','purpose','addr_state','dti','delinq_2yrs','earliest_cr_line','inq_last_6mths','open_acc','pub_rec','revol_bal',\
     'revol_util','total_acc','acc_now_delinq','delinq_amnt','pub_rec_bankruptcies','tax_liens']
data=data[col]
new_col={
'loan_amnt':'贷款金额',
'term':'贷款期限',
'int_rate':'利率',
'installment':'每月还款金额',
'grade':'贷款等级',
'sub_grade':'基础等级',
'emp_length':'工作年限',#（0：少于1年，10：10年及以上）
'home_ownership':'房屋所有权',#(出租、自有、按揭、其他)
'annual_inc':'年收入',
'verification_status':'收入是否由LC验证',
'issue_d':'放款日期',
'loan_status':'target',
'purpose':'贷款目的',
'addr_state':'申请人所在洲',
'dti':'月负债比',
'delinq_2yrs':'过去两年借款人逾期30天以上的数字',
'earliest_cr_line':'信用报告最早日期',
'inq_last_6mths':'过去6个月内被查询次数',
'open_acc':'未还清贷款额度',
'pub_rec':'摧毁公共记录的数量',
'revol_bal':'总贷款金额',
'revol_util':'额度循环使用率',
'total_acc':'总贷款笔数',
'acc_now_delinq':'拖欠的账户数量',
'delinq_amnt':'拖欠的逾期款项',
'pub_rec_bankruptcies':'公开记录破产的数量',
'tax_liens':'留置税数量'}


