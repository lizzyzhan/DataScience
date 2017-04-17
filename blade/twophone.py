# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:36:51 2016
@author: gason
"""

import pandas as pd
import numpy as np
import re
import copy
import matplotlib.pyplot as plt
import report as rpt
from imp import reload
reload(rpt)



data0=pd.read_excel('data.xlsx')
code0=rpt.read_code('code.xlsx')

# 删除第一次推送中手机为中兴的用户
data=data0[(data0['Q1']!=1)|(data0['Q3']!=5)]

# 剔除时间大于1000秒的
data=data[data[u'所用时间']<=1000]
# 剔除2部手机，且答题时间小于60秒的问卷.
data=data[(data['Q1']==1)|(data[u'所用时间']>=60)]




# 设置PPT模板(模板顺序从0开始计数)
mytemplate={'path':'template.pptx','layouts':[2,0]}

'''
for k in code0:
    print('{}: {}'.format(k,code0[k]['content']))
Q1: 推送渠道 
Q2: 您目前在用的手机一共有几部? 
Q3: 请问您当前使用手机的品牌及型号?(如果是多部手机,则填写最主要使用的那一部) 
Q4: 接上题,请问您当时花了多少钱购买该手机?
Q5: 请问您在该部手机上大概的话费套餐是多少? 
Q6: 请问您另外一部正在使用手机的品牌及型号?(如果拥有2部以上手机,则填写其余手机中最主要使用的一部) 
Q7: 接上题,请问您当时花了多少钱购买该手机?
Q8: 请问您该部手机大概的话费套餐是多少? 
Q9: 请问您购买第二部手机的原因？
Q10: 请问您的两部手机在使用上符合下面哪种情况? 
Q11: 性别 
Q12: 年龄 
Q13: 请问您的最高学历? 
Q14: 请问您的职业? 
Q15: 请问选项中哪一个最能代表您个人每月的总收入? 
Q16: 请问您的兴趣爱好?(比较多的花时间也会为此花钱) 
Q17: 接上题,请问您每月会在兴趣爱好上投入占总收入的多少? 
Q18: 请问您平时会关注哪方面的资讯? 
Q19: 请问您的居住地: 
Q20: 非常感谢您的参与!!
'''





#data1=data0[code0['Q1']['qlist']+['Q2','Q3']]
#data2=data0.loc[data0['Q3']==1,data0.columns[17:]]

filename=u'双机市场描述统计_1099'
summary_qlist=None
rpt.summary_chart(data,code0,filename=filename,summary_qlist=summary_qlist,template=mytemplate);               

                                 


               
filename=u'推送方式差异'
cross_qlist=None
cross_class='Q1'
cross_order=['早期自推(384)','自己推送(149)','问卷星代推送(840)']
save_dstyle=None
cresult=rpt.cross_chart(data,code0,cross_class,filename=filename,reverse_display=False,\
cross_qlist=cross_qlist,cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
                 





rcode={2:4,3:4}
data1=data.copy()
data1['Q2'].replace(rcode,inplace=True)
code1=copy.deepcopy(code0)
code1['Q2']['code'][4]='2部及以上'
filename=u'多部手机差异_逆序'
cross_qlist=None
cross_class='Q2'
cross_order=['1部','2部及以上']
save_dstyle=None
cresult=rpt.cross_chart(data1,code1,cross_class,filename=filename,reverse_display=True,\
cross_qlist=cross_qlist,cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)

rcode={4:6,5:6}
data1['Q19c'].replace(rcode,inplace=True)
code1['Q19c']['code'][6]='四线及以下'
filename=u'地域差异分析'
cross_qlist=None
cross_class='Q19c'
cross_order=['北上广深','新一线','二线','三线','四线及以下']
save_dstyle=None
cresult=rpt.cross_chart(data1,code1,cross_class,filename=filename,reverse_display=False,\
cross_qlist=cross_qlist,cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
     



filename=u'男女差异'
cross_qlist=None
cross_class='Q11'
cross_order=['男','女']
save_dstyle=None
cresult=rpt.cross_chart(data0,code0,cross_class,filename=filename,reverse_display=False,\
cross_qlist=cross_qlist,cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)



filename=u'年龄差异'
cross_qlist=None
cross_class='Q12'
cross_order=code0['Q12']['code_order']
save_dstyle=None
cresult=rpt.cross_chart(data0,code0,cross_class,filename=filename,reverse_display=False,\
cross_qlist=cross_qlist,cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)                 
                 




# 双机分布情况


data1=data[data['Q2']!=1]
qq1='Q3'
qq2='Q6'
t=pd.crosstab(data1[qq1],data1[qq2])
t.rename(index=code0[qq1]['code'],columns=code0[qq2]['code'],inplace=True)
'''
t1=t+t.T
for i in t1.index:
    t1.loc[i,i]=t1.loc[i,i]/2

t2=pd.DataFrame(np.tril(t1),index=t1.index,columns=t1.columns)
'''


qlist=['免费赠送的','799元及以下', '800-999元', '1000-1499元', '1500-1999元', '2000-2499元', '2500-2999元', '3000-3499元', '3500元及以上']
t1=pd.DataFrame(t,columns=qlist)


t2=t2.stack()
t2=t2[t2>0]
t2=t2.sort_values(ascending=False)



t=pd.crosstab(data1['Q2'],data1['Q19a'])
    










