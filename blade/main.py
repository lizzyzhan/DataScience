# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:00:26 2017
@author: gason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import report as rpt
reload(rpt)



#  数据d导入
code=rpt.read_code('.\\data\\code.xlsx')
data0=pd.read_excel('.\\data\\data.xlsx',encoding='gbk')

# 数据清洗
data=data0[(data0['Q5']==1)|(data0['Q5'].isnull())]#清楚自己购买但使用不是自己的人
data=data[data[u'来源详情']==u'直接访问']
# 描述统计
#filename=u'小鲜4真实使用用户1_334'
#rpt.summary_chart(data,code,filename=filename)

'''
Q6: 换机原因
Q7: 上一部手机品牌
Q10:关注因素
Q12：选购原因
Q13:手机部数
Q35: 性别
Q36: 年龄
Q37：学历
Q38：职业
Q39：收入
'''


# 交叉统计
cross_class='Q35'
filename=u'性别差异分析'
cross_order=[u'男',u'女',u'总体']
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename,\
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)

#=============================
cross_class='Q36'
filename=u'年龄差异分析'
cross_order=list(code[cross_class]['code_order'])+[u'总体']
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================
cross_class='Q37'
filename=u'学历差异分析'
cross_order=list(code[cross_class]['code_order'])+[u'总体']
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================

cross_class='Q38'
filename=u'职业差异分析'
cross_order=None
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================
cross_class='Q39'
filename=u'收入差异分析'
cross_order=list(code[cross_class]['code_order'])+[u'总体']
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================
cross_class='Q6'
filename=u'换机原因差异分析'
cross_order=None
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================
cross_class='Q7'
filename=u'上一部手机品牌差异分析'
cross_order=None
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================
cross_class='Q10'
filename=u'关注因素差异分析'
cross_order=None
save_dstyle=None
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle)
print(cross_class)
#=============================
