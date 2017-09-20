# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


X=np.random.choice(['a','b','c'],p=[0.2,0.3,0.5],size=1000)
y=np.random.choice(['g','b'],p=[0.7,0.3],size=1000)


def woe_binary(X,y):
    ctable=pd.crosstab(X,y)
    # 如果有0则每一项都加1
    ctable=ctable+1 if (ctable==0).any().any() else ctable
    n_g,n_b=ctable.sum()
    ctable=(ctable/ctable.sum()).assign(woe=lambda x:np.log2(x.iloc[:,0]/x.iloc[:,1]))\
    .assign(ivi=lambda x:(x.iloc[:,0]-x.iloc[:,1])*x['woe'])
    return ctable.loc[:,['woe','ivi']]








