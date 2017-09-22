# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:51:11 2017

@author: 10206913
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import report as rpt
from imp import reload
reload(rpt)

from sklearn import metrics
import sklearn.feature_selection as fs
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from sklearn import preprocessing
from sklearn import datasets



from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


data=pd.read_table('.\\data\\german.data',sep=' ',header=None)
data.columns=['Q{}'.format(i) for i in range(1,21)]+['Y']
Y=data['Y']
X=data.drop(['Y'],axis=1)
N=len(X)


import metric as mm


LabelEncoder=preprocessing.LabelEncoder().fit_transform


#将变量分为因子变量和数值变量分开讨论                 

label_numeric=np.array([np.issubdtype(X.dtypes[i],np.number) for i in range(X.shape[1])])                                                                        
fscore_nonumeric=X.loc[:,~label_numeric].describe().T.assign(missing_pct=lambda x:(N-x['count'])/N)
fscore_numeric=X.loc[:,label_numeric].describe().T.assign(missing_pct=lambda x:(N-x['count'])/N)
for cc in fscore_nonumeric.index:
    fscore_nonumeric.loc[cc,'iv']=mm.info_value(X[cc],Y)
f,p=fs.f_oneway(X.loc[:,label_numeric],Y)
fscore_numeric['f_oneway']=f
fscore_numeric['f_oneway_p']=p


# 测试数据
training=np.random.choice([True,False],p=[0.8,0.2],size=N)

              
              
# 建模
#X.loc[:,~label_numeric]=X.loc[:,~label_numeric].apply(LabelEncoder)
X1=pd.get_dummies(X.loc[:,~label_numeric],drop_first=True)

X2=pd.concat([X1,X.loc[:,label_numeric]],axis=1)
              
rf=RandomForestClassifier(oob_score=True,n_estimators=100)
gbdt=GradientBoostingClassifier()


rf.fit(X2[training],Y[training])
gbdt.fit(X2[training],Y[training])
print(rf.score(X2[~training],Y[~training]))
print(gbdt.score(X2[~training],Y[~training]))


data=pd.read_csv('.\\data\\LoanStats_2017Q1.csv',skiprows=0,header=1)
set(data.dtypes)

summary=data.select_dtypes(include=['O']).describe().T.assign(missing_pct=data.apply(lambda x:(len(x)-x.count())/len(x)))



