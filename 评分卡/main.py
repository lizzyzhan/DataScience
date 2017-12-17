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

data=pd.read_csv('LoanStats_2016Q4.csv',skiprows=1)
data.head()
data0=data
#import missingno
#missingno.matrix(data)


check_null=data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data))
check_null[check_null>0]

data=data.dropna(thresh=len(data)*0.5,axis=1)
data=data.loc[:,data.apply(pd.Series.nunique)!=1]
len(set(list(data.columns)))==data.shape[1]
data.shape


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

data=data.rename(columns=new_col)

# 处理目标变量
data['target']=data['target'].replace({'Current':np.nan,'Fully Paid':0,'Charged Off':1,'Late (31-120 days)':1,'In Grace Period':np.nan,\
                                      'Late (16-30 days)':1,'Default':np.nan})
data=data.loc[data['target'].notnull(),:]
data['放款日期']=pd.to_datetime(data['放款日期'])
data['信用报告最早日期']=pd.to_datetime(data['信用报告最早日期'])
data['days']=(data['放款日期']-data['信用报告最早日期']).map(lambda x:x.days)

data=data.drop(['放款日期','信用报告最早日期'],axis=1)

var_list,data=rpt.analysis.var_detection(data)
type_of_var=dict(zip(data.columns,[s['vtype'] for s in var_list]))
categorical_var=[]
continuous_var=[]
for k,v in type_of_var.items():
    if v in ['category']:
        categorical_var.append(k)
    elif v == 'number':
        continuous_var.append(k)

categorical_var.remove('target')

# 补缺
for v in continuous_var:
    s=list(data[v].dropna().unique())
    data[v]=data[v].map(lambda x : np.random.choice(s,1)[0] if '%s'%x=='nan' else x)

for v in categorical_var:
    s=data[v].value_counts().argmax()
    data[v]=data[v].map(lambda x : s if '%s'%x=='nan' else x)




#rpt.analysis.var_detection(data)
#rpt.AnalysisReport(data,'club report')
#var_list,data=rpt.analysis.var_detection(data)

#type_of_var=dict(zip(data.columns,[s['vtype'] for s in var_list]))

from metrics import WeightOfEvidence
from discretization import Discretization

data0=data.copy()
#WeightOfEvidence()
max_intervals=6


N=len(data)
woe_iv={}
print('=============处理因子变量==================')
for v in categorical_var:
    print(v)
    if not(isinstance(data[v].iloc[0],str)) and len(data[v].unique())>max_intervals:
        #print('进行分箱操作')
        dis=Discretization(method='chimerge',max_intervals=max_intervals)
        dis.fit(data[v],data['target'])
        data.loc[:,v]=dis.transform(data[v])
    woe=WeightOfEvidence()
    woe.fit(data[v],data['target'])
    woe_iv[v]={'woe':woe.woe,'iv':woe.iv}
    data[v]=data[v].replace(woe.woe).astype(np.float64)

print('===============处理连续变量================')
for v in continuous_var:
    print(v)
    if len(data[v].unique())>max_intervals:
        #print('进行分箱操作')
        dis=Discretization(method='chimerge',max_intervals=max_intervals,sample=1000)
        dis.fit(data[v],data['target'])
        data.loc[:,v]=dis.transform(data[v])
    woe=WeightOfEvidence()
    woe.fit(data[v],data['target'])
    woe_iv[v]={'woe':woe.woe,'iv':woe.iv}
    data[v]=data[v].replace(woe.woe).astype(np.float64)


iv_threshould=0.01
varByIV=[k for k,v in woe_iv.items() if v['iv'] > iv_threshould]
print('剩余%d个变量'%len(varByIV))
print(varByIV)

# 此处还可以进行多变量分析，共线性分析等方法来去除变量
var_WOE_list=varByIV
X=data[var_WOE_list]
y=data['target']

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


'''逻辑回归'''
from sklearn.cross_validation import train_test_split as sp
from sklearn.linear_model import LogisticRegression as LR

X=data[features_selection]
y=data['traget']
X_train,X_test,y_train,y_test=sp(X,y,test_size=0.3,random_stated=1)

lr=LR()
lr.fit(X_train,y_train)
y_train_label=lr.predict(X_train)
y_test_label=lr.predict(X_test)

from sklearn.metrics import accuracy_score
print('训练集准确率：{:.2%}'.format(accuracy_score(y_train_label,y_train)))
print('测试集准确率：{:.2%}'.format(accuracy_score(y_test_label,y_test)))



from sklearn.cross_validation import train_test_split as sp
X_train,X_test,y_train,y_test=sp(X,y,test_size=0.3,random_stated=1)

modelfit(xgb1,X_train,y_train)


