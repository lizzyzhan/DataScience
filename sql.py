# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:10:28 2017

@author: 10206913
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import report as rpt
from imp import reload
reload(rpt)



qid='blade_v8'
userid_begin=10000
code=rpt.read_code('code.xlsx')
data=pd.read_excel('data.xlsx')
data,code=rpt.to_dummpy(data,code,qtype_new='单选题')
N=len(data)
data.index=[userid_begin+i+1 for i in range(N)]
t=data.stack().reset_index()
t.columns=['userid','qn_an','code']
t['qnum']=t['qn_an'].map(lambda x:x.split('_')[0])
t['itemnum']=t['qn_an'].map(lambda x:'_'.join(x.split('_')[1:]))
t['qtype']=t['qnum'].map(lambda x: code[x]['qtype'])
t['qname']=t['qnum'].map(lambda x: code[x]['content'])
t['itemname']=t['qn_an'].map(lambda x: code[x.split('_')[0]]['code_r'][x] if \
 code[x.split('_')[0]]['qtype']=='矩阵单选题' else code[x.split('_')[0]]['code'][x])



'''
t['value']=''
for index in t.index:
    if t.loc[index,'qtype']=='矩阵单选题':
        if t.loc[index,'code']==0:
            t.loc[index,'value']='否'
        else:
            t.loc[index,'value']=code[t.loc[index,'qnum']]['code'][t.loc[index,'code']]
    elif t.loc[index,'qtype']=='排序题':
        t.loc[index,'value']='Top{}'.format(t.loc[index,'code'])
    else:
        t.loc[index,'value']='是' if t.loc[index,'code']==1 else '否'
'''

 
t['value']=''
t['tmp']=t['qnum']+t['code'].map(lambda x:'_%s'%int(x)) 
tmp1=t.loc[t['code']>0,'tmp'].map(lambda x: code[x.split('_')[0]]['code'][int(x.split('_')[1])] if code[x.split('_')[0]]['qtype']=='矩阵单选题' else '')

t['value']=t.loc[t['code']>0,'tmp'].map(lambda x: 'Top{}'.format(x.split('_')[1]) if code[x.split('_')[0]]['qtype']=='排序题' else '')
t.loc[t['code']==0,'value']='否' 
t.loc[t['code']==1,'value']='是' 
    
t['qid']=qid     
qdata=pd.DataFrame(t,columns=['userid','qid','qnum','qname','qtype','itemnum','itemname','code','value'])
       
      




