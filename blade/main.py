# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:00:26 2017
@author: gason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import report as rpt
from imp import reload
reload(rpt)


# 设置PPT模板(模板顺序从0开始计数)
mytemplate={'path':'template.pptx','layouts':[2,0]}

#  数据d导入
code=rpt.read_code('.\\data\\code.xlsx')
data0=pd.read_excel('.\\data\\data.xlsx',encoding='gbk')

# 数据清洗
data=data0[(data0['Q5']==1)|(data0['Q5'].isnull())]#清楚自己购买但使用不是自己的人
data=data[data[u'来源详情']==u'直接访问']
data['Q40c'].replace({4:6,5:6},inplace=True)



# 描述统计
filename=u'小鲜4真实使用用户1_334'
rpt.summary_chart(data,code,filename=filename,template=mytemplate)

'''
Q1	请问您的 “小鲜4”符合下面哪种情况?
Q2	是否有签约运营商的套餐？
Q3	签约套餐
Q4	请问您签约套餐后，对应的购机费用是多少？
Q5	请问您购买的“小鲜4”是谁在使用?
Q6	请问您当初想要换手机的主要原因是什么?
Q7	请问您使用的上一部手机(“小鲜4”之前)的品牌及型号?
Q7a	上一部手机的型号
Q8	您在购买前,是否有和其他手机做过对比?
Q9	您和哪些手机品牌做过对比? 型号是什么?
Q10	请问您在购买手机时,最关注的前几项因素是哪些?
Q11	请问在购买前,您从哪里了解到“小鲜4”?
Q12	请问您当初选购“小鲜4”最主要的原因是什么?
Q13	您目前在用的手机一共有几部?
Q14	请问您其他手机的品牌和型号是什么?
Q15	请问“小鲜4”是不是您最主要使用的手机?
Q16	拥有各个运营商的手机号数
Q16b	小鲜4对应手机号的运营商
Q17	请问您平均每个月在“小鲜4”手机号（收到问卷的手机号）上产生了多少资费？
Q18	请问您平均每个月在“小鲜4”手机号（收到问卷的手机号）上产生了多少流量？
Q19	平均每个月在各个运营商上产生的资费
Q20	平均每个月在各个运营商上产生的流量
Q21	请问你平时使用“小鲜4”手机最多的5个功能是什么,请帮忙排序?
Q22	智慧语音使用情况
Q23	请问您每天大概花多长时间在使用“小鲜4”?
Q24	各个功能每日使用时长
Q25	请问您为“小鲜4”单独买了哪些配件?
Q26	请问您了解VR/AR吗？请选择最符合您的选项。
Q27	模块满意度
Q28	您对“小鲜4”整体的满意度如何?(以0-10分为标准,0分为最低,10分为最高)
Q29	【接上题】请写下您给出上述分数的原因?以便我们持续改进、提升产品的用户体验。
Q30	您有多大意愿向您身边的家人或朋友推荐这款“小鲜4"?(0分表示完全不想推荐,10分表示非常愿意推荐)
Q31	【接上题】请写下您给出上述分数的原因?以便我们持续改进、提升产品的用户体验。
Q32	下面是一些关于手机的描述语句,请选择其中与您相符合的选项。
Q33	消费态度
Q34	下面是一些关于科技产品的描述语句,请选择最符合您自身情况的一个。
Q35	请问您的性别？
Q36	请问您的年龄?
Q37	请问您的最高学历是?
Q38	请问您的职业是?
Q39	请问选项中哪一个最能代表您个人每月的总收入呢?
Q40	请选择您所在的城市:
Q40a	省份
Q40b	城市
Q40c	城市划分
Q43	您是否愿意接受我们面对面的访谈?我们希望能更进一步了解您使用“小鲜4”的情况,接受访谈我们会有更丰厚的礼品作为回报。
'''


# 交叉统计
cross_class='Q35'
filename=u'性别差异分析'
cross_order=[u'男',u'女']
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename,\
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)

#=============================
cross_class='Q36'
filename=u'年龄差异分析'
cross_order=list(code[cross_class]['code_order'])
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
cross_class='Q37'
filename=u'学历差异分析'
cross_order=list(code[cross_class]['code_order'])
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
cross_class='Q38'
filename=u'职业差异分析'
cross_order=None
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
cross_class='Q39'
filename=u'收入差异分析'
cross_order=list(code[cross_class]['code_order'])
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
cross_class='Q40c'
filename=u'城市差异分析'
cross_order=list(code[cross_class]['code_order'])
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================

'''
cross_class='Q6'
filename=u'换机原因差异分析'
cross_order=None
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
cross_class='Q7'
filename=u'上一部手机品牌差异分析'
cross_order=None
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
cross_class='Q10'
filename=u'关注因素差异分析'
cross_order=None
save_dstyle=['CHI']
rpt.cross_chart(data,code,cross_class,filename=filename, \
cross_order=cross_order,save_dstyle=save_dstyle,template=mytemplate)
print(cross_class)
#=============================
'''