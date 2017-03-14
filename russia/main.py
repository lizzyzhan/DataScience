# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:27:54 2017

@author: 10206913
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:36:51 2016
@author: gason
"""

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import report as rpt
from imp import reload
reload(rpt)



#t2=pd.DataFrame(index=t1.index,columns=['40kRUB','40-60RUB','60RUB-80RUB','80-100RUB','100RUB'])

# =================[数据导入]==========================
russia=pd.read_excel('.\\data\\data_russia.xlsx')
germany=pd.read_excel('.\\data\\data_germany.xlsx')
spain=pd.read_excel('.\\data\\data_spain.xlsx')
code=rpt.read_code('.\\data\\code_cn.xlsx')


'''
Q48: 技术产品的态度
Q47: 个人消费观的态度题
Q46: 关于手机本身的态度
Q24: 购买关注因素
Q25: 卖点探究 (矩阵单选题，额外多付钱）
Q26: 购买途径
Q3: 性别
Q5: 年龄
Q8: 使用的电子产品
Q9: 手机部数
Q10: What's your role when purchasing your phone? Please answer following questions based on the mobile phone you currently use most often. 
Q11: 当前手机使用时长
Q53: 购买动机（换机原因）
Q23: 了解途径
Q27: 手机价格
Q28: 换机时长
Q29: 下一部手机价格
Q32: 当前手机不满意的点
Q49: 婚姻
Q50: 职业
Q51: 学历
Q52: 个人收入
'''


# 俄罗斯
#data['Q27'].replace({60:63,61:63,62:63},inplace=True)
t=rpt.qtable(russia,code,'Q24','Q27')['fo']
# 55,56,57,58,59,60,61,62,63
code_order=['5000RUB以下','5000-9999RUB','10000-14999RUB','15000-19999RUB',\
'20000-24999RUB','25000-29999RUB','30000-34999RUB','35000-39999RUB','40000RUB以上']
t=pd.DataFrame(t,columns=code_order)
c=rpt.mca(t,2)[1]
from sklearn import metrics
dd=metrics.pairwise.cosine_distances(c)
dd=pd.DataFrame(dd,index=c.index,columns=c.index)
dd=np.arccos(1-dd)*180/np.pi


rcode={55:56,58:59,60:63,61:63,62:63}
code1=code.copy()
code1['Q27']['code'][56]='10000RUB以下'
code1['Q27']['code'][59]='15000-24999RUB'
code1['Q27']['code'][63]='25000RUB以上'
t=rpt.qtable(russia.replace(rcode),code1,'Q24','Q27')['fo']
# 55,56,57,58,59,60,61,62,63
code_order=['10000RUB以下','10000-14999RUB','15000-24999RUB','25000RUB以上']
t=pd.DataFrame(t,columns=code_order)
r,c,m=rpt.mca(t,2)

w=pd.ExcelWriter(u'mca_russia.xlsx')
r.to_excel(w,startrow=0,index_label=True)
c.to_excel(w,startrow=len(r)+2,index_label=True)
w.save()

































# 设置PPT模板(模板顺序从0开始计数)
mytemplate={'path':'template.pptx','layouts':[2,0]}