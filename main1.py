import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import report as rpt
reload(rpt)



#  数据d导入
code=rpt.read_code('.\\data\\code.xlsx')
data0=pd.read_excel('.\\data\\data.xlsx',encoding='gbk')

# 数据清晰
data=data0[(data0['Q5']==1)|(data0['Q5'].isnull())]#清楚自己购买但使用不是自己的人
data=data[data[u'来源详情']==u'直接访问']

'''
Q12=data[code['Q12']['qlist']]
Q12.applymap(lambda x:int(x==1))
Q12=Q12.sum()
Q12.rename(index=code['Q12']['code'],inplace=True)
Q12.sort_values(inplace=True)
'''

filename=u'小鲜4真实使用用户1_334'
rpt.summary_chart(data,code,filename=filename)
