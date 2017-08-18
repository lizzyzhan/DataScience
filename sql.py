# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:44:07 2017

@author: 10206913
"""

import pandas as pd
import report as rpt
from imp import reload
reload(rpt)
import os

'''
dirlist=os.listdir()
userid_begin=1000000
qdata=pd.DataFrame()
quesinfo=pd.DataFrame()
NN=0
for dirname in dirlist:
    if not os.path.isdir(dirname):
        continue
    print(dirname)
    if os.path.exists(os.path.join(dirname,'data.xlsx')) and os.path.exists(os.path.join(dirname,'code.xlsx')):
        code=rpt.read_code(os.path.join(dirname,'code.xlsx'))
        data=rpt.read_data(os.path.join(dirname,'data.xlsx'))
    else:
        data,code=rpt.wenjuanxing(dirname)
    NN+=len(data)
    print('该数据{}份，累计{}份'.format(len(data),NN))
    qdata0,quesinfo0=rpt.qdata_flatten(data,code,quesid=dirname,userid_begin=userid_begin)
    qdata=pd.concat([qdata,qdata0],axis=0,ignore_index=True)
    quesinfo=pd.concat([quesinfo,quesinfo0],axis=0,ignore_index=True)
    userid_begin+=len(data)
'''

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
quesinfo=pd.read_csv(u'quesinfo.csv')
qdata=pd.read_csv(u'qdata.csv')

quesinfo=quesinfo.rename(columns={'percent(%)':'percent'})


qdata['UID']=range(len(qdata))
d=pysqldf('select distinct(quesid) from quesinfo')


print(pysqldf("select quesid,qnum,itemname,percent from quesinfo where itemname=='男' and qnum like '%关注%'"))






s="select a.qnum,a.itemnum,b.qnum,b.itemnum, count(a.code) from qdata as a,qdata as b where \
a.userid ==b.userid and a.quesid =='A452_russia' and b.quesid=='A452_russia' and a.qnum=='Q2' \
and b.qnum=='Q3' and a.code==1 and b.code==1 group by a.qnum,a.itemnum,b.qnum,b.itemnum"

print(pysqldf(s))

