# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:44:43 2018

@author: gason
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import pywt

df = pd.read_excel('./data/by_day_consign_20150101_20180114.xlsx')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
#df['y'] = np.log(df['y'])



(cA, cD) = pywt.dwt(df['y'], 'db4')
n = len(df)
cAn = pywt.upcoef('a', cA, 'db4', take=n)
cDn = pywt.upcoef('a', cD, 'db4', take=n)
fig,[ax0,ax1,ax2] = plt.subplots(3,1)
ax0.plot(df['ds'],df['y'],'.')
ax0.set_xlabel('Raw')

ax1.plot(df['ds'],cAn,'.')
ax1.set_xlabel('Low Frequency')

ax2.plot(df['ds'],cDn,'.')
ax2.set_xlabel('High Frequency')
fig.tight_layout() #fig.savefig
# df['y'] = cAn
df['y'] = cDn













m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=45)
forecast = m.predict(future)



forecast['yhat_exp'] = np.exp(forecast['yhat'])
result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper','yhat_exp']].tail(45)
result.to_csv('pred5.csv')











# ================================================================
# https://facebook.github.io/prophet/docs/diagnostics.html
# https://github.com/facebook/prophet/blob/v0.3/notebooks/diagnostics.ipynb

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from sklearn.grid_search import ParameterGrid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


holiday=pd.read_excel('.\\data\\holiday_info.xlsx')

df = pd.read_excel('./data/by_day_consign_20150101_20180114.xlsx')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df['yraw']=df['y']
df['y'] = np.log(df['y'])



param_grid={'growth':['linear']
,'seasonality_prior_scale':[1,3,5,8,10,15]
,'holidays_prior_scale':[10,30,50,70,90,110,130]
#,'changepoint_prior_scale':[0.01,0.03,0.05,0.08,0.1,0.2]
#,'interval_width':[0.2,0.4,0.6,0.8]
}


param_grid={'growth':['linear']
,'seasonality_prior_scale':[1,2,3]
,'holidays_prior_scale':[100,120,130,140,150,160,170,180]
#,'changepoint_prior_scale':[0.01,0.03,0.05,0.08,0.1,0.2]
#,'interval_width':[0.2,0.4,0.6,0.8]
}


param_list=list(ParameterGrid(param_grid))
#scores=pd.DataFrame(columns=['seasonality','holidays','mape','rmse'])
scores=[]
for i,param in enumerate(param_list):
    print('{}/{}:'.format(i,len(param_list)),param)
    m=Prophet(holidays=holiday,**param)
    m.fit(df)   
    df_cv = cross_validation(m, horizon='{} days'.format(30), period='100 days', initial='730 days')
    df_cv['yhat']=np.exp(df_cv['yhat'])
    df_cv['y']=np.exp(df_cv['y'])
    mape=np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y']))
    rmse=np.sqrt(np.mean((df_cv['y'] - df_cv['yhat'])**2))
    scores.append([param['seasonality_prior_scale'],param['holidays_prior_scale'],mape,rmse])
    print(len(scores),' : ',mape)


scores=pd.DataFrame(scores)
scores.columns=['seasonality','holidays','mape','rmse']
print('最小的mape是：{:.2f} %',100*scores['mape'].min())
print('对应的参数是：{}={}, {}={}'.format(
'seasonality_prior_scale',scores.loc[scores['mape'].argmin(),'seasonality']
,'holidays_prior_scale',scores.loc[scores['mape'].argmin(),'holidays']))
# mape = 11.20%,  seasonality_prior_scale=1,holidays_prior_scale=130


param={'growth':'linear','seasonality_prior_scale':1,'holidays_prior_scale':130}
m=Prophet(holidays=holiday,**param)
m.fit(df)
forcasts=m.predict(df[['ds']])


fig,ax=plt.subplots()
ax.plot(df['ds'],np.exp(df['y']),'.',alpha=0.4,color='b')
ax.plot(df['ds'],np.exp(forcasts['yhat']),'-',color='r')




