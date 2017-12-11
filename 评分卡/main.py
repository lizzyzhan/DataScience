# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:14:04 2017

@author: gason
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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

from imp import reload
import reportgen as rpt
reload(rpt)
rpt.AnalysisReport(data,'club report')