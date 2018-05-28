# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:44:43 2018

@author: gason
"""

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
sns.set()
#import pywt


holidays=pd.read_excel('.\\data\\holidays.xlsx')
holidays=holidays.loc[holidays['holiday'].notnull(),['ds','holiday']].reset_index(drop=True)



df = pd.read_excel('./data/tehr.xlsx')
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = np.log(df['y'])


## 默认参数
#m=Prophet(holidays=holidays)
#m.fit(df)
##对历史数据进行预测，预测值对应着yhat列
#forcasts=m.predict(df[['ds']])
## 预测未来30天数据
#future=m.make_future_dataframe(periods=30)
#forecast=m.predict(future)
##plot
#m.plot(forecast);
#m.plot_components(forecast,weekly_start=1)


# 利用交叉验证调参

param_grid={'growth':['linear']
,'seasonality_prior_scale':[5,8,10,20,30,50,100]
,'holidays_prior_scale':[5,8,10,20,30,50,100]
#,'changepoint_prior_scale':[0.01,0.03,0.05,0.08,0.1,0.2]
#,'interval_width':[0.2,0.4,0.6,0.8]
}


param_list=list(ParameterGrid(param_grid))
#scores=pd.DataFrame(columns=['seasonality','holidays','mape','rmse'])

scores=[]
for i,param in enumerate(param_list):
    print('{}/{}:'.format(i,len(param_list)),param)
    m=Prophet(holidays=holidays,**param)
    m.fit(df)   
    df_cv = cross_validation(m, horizon='{} days'.format(30), period='120 days', initial='1095 days')
    df_cv['yhat']=np.exp(df_cv['yhat'])
    df_cv['y']=np.exp(df_cv['y'])
    mape=np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y']))
    rmse=np.sqrt(np.mean((df_cv['y'] - df_cv['yhat'])**2))
    scores.append([param['seasonality_prior_scale'],param['holidays_prior_scale'],mape,rmse])
    print('MAPE : {:.5f}%'.format(100*mape))


scores=pd.DataFrame(scores)
scores.columns=['seasonality','holidays','mape','rmse']
print('最小的mape是：{:.2f} %'.format(100*scores['mape'].min()))
print('对应的参数是：{}={}, {}={}'.format(
'seasonality_prior_scale',scores.loc[scores['mape'].argmin(),'seasonality']
,'holidays_prior_scale',scores.loc[scores['mape'].argmin(),'holidays']))
# 测试集的mape = 8.83%,  优化后的参数：seasonality_prior_scale=50,holidays_prior_scale=20




# 绘图，查看调参后的模型在历史数据上的结果
param={'growth':'linear','seasonality_prior_scale':scores.loc[scores['mape'].argmin(),'seasonality'],\
'holidays_prior_scale':scores.loc[scores['mape'].argmin(),'holidays']}
param={'growth':'linear','seasonality_prior_scale':50,'holidays_prior_scale':20}
m=Prophet(holidays=holidays,**param)
m.fit(df)
forecasts=m.predict(df[['ds']])
forecasts['y']=df['y']

fig,ax=plt.subplots()
ax.plot(df['ds'],df['y'],'.',alpha=0.4,color='b')
ax.plot(df['ds'],forecasts['yhat'],'-',alpha=0.6,color='b')
fig.savefig('.\\_images\\prophet_forecast.png',dpi=500)


# decompose
fig,axs=plt.subplots(4,1)
axs[0].plot(forecasts['ds'],forecasts['y'])
axs[0].set_ylabel('Raw')
axs[1].plot(forecasts['ds'],forecasts['trend'])
axs[1].set_ylabel('Trend')
axs[2].plot(forecasts['ds'],forecasts['seasonal'])
axs[2].set_ylabel('seasonal')
axs[3].plot(forecasts['ds'],forecasts['y']-forecasts['yhat'])
axs[3].set_ylabel('residual')
fig.savefig('.\\_images\\prophet_decompose.png',dpi=500)







