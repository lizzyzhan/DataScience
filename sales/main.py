# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:59:01 2017

@author: 10206913
"""

import  pandas as pd
import re

data0=pd.read_excel('russia_sales.xlsx')
data00=pd.read_excel('rusia_sales_W16Y16_W05Y17.xlsx')
data0=data0.append(data00,ignore_index=True)

fields=['Product_name',
 'Product',
 'Product_series',
 'BRAND',
 'Company',
 'FirstActivity',
 'GENERATION TOTAL*',
 'Productgroup',
 'DESIGN',
 'MEMORY CARD SLO',
 'CAMERA',
 'METAL CASE',
 'Resolution in Ths.Pixel*',
 'Display Resolution in Pixel',
 'NO.OF SIM CARD',
 'KEYBOARD',
 'OPERATING SYST.',
 'PROCESSOR BRAND',
 'CSP-MHZ',
 'CPU-CORES',
 'DISPLAY SIZE',
 'update_time',
 'Year',
 'Week',
 'Country',
 'Sales Units',
 'Price EUR/Un.',
 'PRICE USD/UN.',
 'PRICE RUB/UN.',
 'Sales Units%',
 'SALES RUB',
 'SALES USD',
 'Sales Value%']


data0['DISPLAY SIZE']=data0['DISPLAY SIZE'].map(lambda x:'%s'%x)
data0['GENERATION TOTAL*']=data0['GENERATION TOTAL*'].map(lambda s:re.sub('X','x',s))
ind1=data0['GENERATION TOTAL*']=='4.x G'
ind2=data0['OPERATING SYST.']=='ANDROID'
ind3=data0['DISPLAY SIZE']>='4.7'
ind4=data0['DISPLAY SIZE']<='6'
data=data0[ind1&ind2&ind3&ind4]

'''
W42Y16更新的8周，共429款满足条件
W05Y17更新的13周，共649款满足条件

'''

data['fweek']=data['Year'].map(lambda x:'%s'%(x-2000))+'W'+data['Week'].map(lambda x:'%.2d'%x)

# 销售量
t1=data.groupby(['fweek','DISPLAY SIZE'])['Sales Units'].sum()
t1=t1.unstack()
t1.drop(['4.95','4.9','5.9','5.6'],axis=1,inplace=True)
t1=t1.T.sort_values(by='17W05',ascending=False)
t1.T.to_excel('销量_频数.xlsx')
tt1=t1/t1.T.sum(axis=1)
tt1=tt1.T
tt1.fillna(0,inplace=True)
tt1.to_excel('销量_占比.xlsx')


# 销售额
t3=data.groupby(['fweek','DISPLAY SIZE'])['SALES RUB'].sum()
t3=t3.unstack()
t3.drop(['4.95','4.9','5.9','5.6'],axis=1,inplace=True)
t3=t3.T.sort_values(by='17W05',ascending=False)
t3.T.to_excel('销售额_频数.xlsx')
tt3=t3/t3.T.sum(axis=1)
tt3=tt3.T
tt3.fillna(0,inplace=True)
tt3.to_excel('销售额_占比.xlsx')


'''
5       214.0
5.5     102.0
5.2      47.0
4.7      19.0
6        14.0
5.7      12.0
5.1       9.0
4.8       4.0
5.3       4.0
4.93      1.0
5.4       1.0
4.95      1.0
5.6       1.0

'''












'''
# 按照时间来计算
m=data['FirstActivity'].map(lambda x: x.split(' ')[0])
quarter=m.replace({'January':'Q1','February':'Q1','March':'Q1',\
           'April':'Q2','May':'Q2','June':'Q2',\
           'July':'Q3','August':'Q3','September':'Q3',\
           'October':'Q4','November':'Q4','December':'Q4'})
y=data['FirstActivity'].map(lambda x: x.split(' ')[1])
data['Activity']=y+quarter
data_phone=data[data['Week']==42]
t2=data_phone.groupby(['DISPLAY SIZE','Activity'])['DISPLAY SIZE'].count()
t2=t2.unstack()
'''

