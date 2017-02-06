# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:51:31 2017

@author: gason
"""
import report as rpt
import pandas as pd
filepath1=['.\\data\\','.\\data\\']
data1,code1=rpt.wenjuanxing(filepath1)
data1['Q0']=1
d1=len(data1)

filepath2=['.\\data\\','.\\data\\']
data2,code2=rpt.wenjuanxing(filepath2)
data2['Q0']=2
d2=len(data2)

filepath3=['.\\data\\','.\\data\\']
data3,code3=rpt.wenjuanxing(filepath3)
data3['Q0']=3
d3=len(data3)

qn=12
qlist=['Q%d'%(i) for i in range(qn)]
data=pd.DataFrame(columns=qlist,index=[i+1 for i in range(d1+d2+d3)])
code={}
ind={'Q0':['Q0','Q0','Q0',code1['Q0']],\
'Q1':[],\
'Q2':[],\
'Q3':[],\
'Q4':[],\
'Q5':[],\
'Q6':[],\
'Q7':[],\
'Q8':[],\
'Q9':[],\
'Q10':[],\
'Q11':[],\
'Q12':[],\
'Q13':[],\
'Q14':[],\
'Q15':[],\
'Q16':[],\
}

for qq in qlist:
    if ind[qq][0]:
        data.loc[[i+1 for i in range(d1)],qq]=list(data1[ind[qq][0]])
    if ind[qq][1]:
        data.loc[[d1+i+1 for i in range(d2)],qq]=list(data2[ind[qq][1]])
    if ind[qq][2]:
        data.loc[[d1+d2+i+1 for i in range(d3)],qq]=list(data3[ind[qq][2]])
    code[qq]=ind[qq][3]



