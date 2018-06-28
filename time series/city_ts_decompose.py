# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 19:20:17 2018

@author: gason
"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler


# 节假日数据导入
hld=pd.read_excel('.\\data\\holidays.xlsx')
#hld['bahld']=None
#for i in hld.index:
#    cond1=hld.loc[i,'daysbeforeholiday']<0 and hld.loc[i,'daysbeforeholiday']+hld.loc[i,'daysafterholiday']>=0
#    cond2=hld.loc[i,'daysbeforeholiday']<0 and hld.loc[i,'daysbeforeholiday']+hld.loc[i,'daysafterholiday']<0
#    cond3=hld.loc[i,'daysbeforeholiday']>=0 and hld.loc[i,'daysbeforeholiday']+hld.loc[i,'daysafterholiday']<0
#    cond4=hld.loc[i,'daysbeforeholiday']>=0 and hld.loc[i,'daysbeforeholiday']+hld.loc[i,'daysafterholiday']>0
#    cond5=hld.loc[i,'daysbeforeholiday']>=0 and hld.loc[i,'daysbeforeholiday']+hld.loc[i,'daysafterholiday']==0
#    if cond1:
#        hld.loc[i,'bahld']='BH'+str(np.abs(hld.loc[i,'daysbeforeholiday'])).zfill(2)
#    elif cond2:
#        hld.loc[i,'bahld']='AH'+str(np.abs(hld.loc[i,'daysafterholiday'])).zfill(2)
#    elif cond3:
#        hld.loc[i,'bahld']='HA'+str(np.abs(hld.loc[i,'daysbeforeholiday'])+1).zfill(2)
#    elif cond4:
#        hld.loc[i,'bahld']='HB'+str(np.abs(hld.loc[i,'daysafterholiday'])+1).zfill(2)
#    elif cond5:
#        hld.loc[i,'bahld']='HAB'
#    else:
#        hld.loc[i,'bahld']=''



# 间夜量数据导入
# data 的列： ds,y,cityid,cityname
df = pd.read_excel('./data/tehr_city.xlsx')
df['ds'] = pd.to_datetime(df['ds'])
df=df.loc[df['ds']<='2018-06-25',:]
df.loc[df['y']==0,'y']=None
df['y']=np.log(df['y'])
df=df.merge(hld[['ds','holiday','bahld']],how='left',on='ds')

dim_city=df.groupby(['cityid','cityname'])['y'].sum().rank(ascending=False).reset_index().rename(columns={'y':'rank'}).sort_values('rank')




# 设定 prophet 的节假日参数
holidays=hld.loc[hld['bahld'].isin(['HA01','HA02','HA03','HA04','HAB','HB04','HB03','HB02']),:].reset_index(drop=True)
holidays['lower_window']=0
holidays['upper_window']=0
holidays.loc[holidays['bahld']=='HA01','lower_window']=-4
holidays.loc[holidays['bahld']=='HA02','lower_window']=-1
holidays.loc[holidays['bahld']=='HB02','upper_window']=2
holidays=holidays[['ds','holiday','lower_window','upper_window']]


def ts_evaluation(df,param,horizon=30,period=120,initial=1095,exp=True):
    '''
    利用交叉验证评估效果   
    '''

    #param={'holidays':holidays,'growth':'linear','seasonality_prior_scale':50,'holidays_prior_scale':20}
    m=Prophet(**param)
    m.fit(df) 
    forecasts=m.predict(df[['ds']])
    forecasts['y']=df['y']
    df_cv = cross_validation(m, horizon='{} days'.format(30), period='{} days'.format(period), initial='{} days'.format(initial))
    if exp:
        df_cv['yhat']=np.exp(df_cv['yhat'])
        df_cv['y']=np.exp(df_cv['y'])
    mape=np.mean(np.abs((df_cv['y'] - df_cv['yhat']) / df_cv['y']))
    rmse=np.sqrt(np.mean((df_cv['y'] - df_cv['yhat'])**2))
    scores={'mape':mape,'rmse':rmse}
    return scores



# 利用交叉验证调参


def ts_grid_search(df,holidays,param_grid=None,cv_param=None,RandomizedSearch=True,random_state=None):
    '''网格搜索
    时间序列需要特殊的交叉验证
    
    df:   
    holidays: 需要实现调好  
    
    '''

    df=df.copy()
    if param_grid is None:
        param_grid={'growth':['linear']
        ,'seasonality_prior_scale':np.round(np.logspace(0,2.2,10))
        ,'holidays_prior_scale':np.round(np.logspace(0,2.2,10))
        ,'changepoint_prior_scale':[0.05] #[0.005,0.01,0.02,0.03,0.05,0.008,0.10,0.13,0.16,0.2]
        ,'interval_width':[0.80] #[0.2,0.4,0.6,0.8]
        }


    if RandomizedSearch:
        param_list=list(ParameterSampler(param_grid,n_iter=10,random_state=random_state))
    else:
        param_list=list(ParameterGrid(param_grid))
    
    if cv_param is None:
        cv_param={'horizon':30,'period':120,'initial':1095}


    scores=[]
    for i,param in enumerate(param_list):
        print('{}/{}:'.format(i,len(param_list)),param)
        param.update({'holidays':holidays})
        scores_tmp=ts_evaluation(df,param,exp=True,**cv_param)        
        param.pop('holidays')
        tmp=param.copy()
        tmp.update({'mape':scores_tmp['mape'],'rmse':scores_tmp['rmse']})
        scores.append(tmp)                       
        print('mape : {:.5f}%'.format(100*scores_tmp['mape']))


    scores=pd.DataFrame(scores)
    
    best_param_=scores.loc[scores['mape'].argmin(),:].to_dict()
    best_scores_=best_param_['mape']
    best_param_.pop('mape')
    best_param_.pop('rmse')
   

    return best_param_,best_scores_,scores
    



def ts_decompose(forecasts):
    '''
    时间序列相关成分分解
    
    '''
    
    assert set(['ds','y','residual','holiday','bahld']).issubset(set(forecasts.columns)) 

                       
    tsdecompose={}
    
    t0=np.sum(forecasts['seasonal']**2)
    tsdecompose['seasonal']=t0/np.sum(forecasts['yhat']**2)
    for cc in ['yearly','weekly','holidays','residual']:
        t=np.sum(forecasts[cc]**2)
        tsdecompose[cc]=t/t0
    
    
    days=pd.date_range(start='{}-01-01'.format(forecasts['ds'].max().year-1), periods=365)
    assert days.isin(forecasts['ds']).all()
    yforecasts=forecasts.loc[forecasts['ds'].isin(days),:].reset_index(drop=True)
    
    # 添加季度数据
    yforecasts=yforecasts.assign(quarter=lambda x:'Q')
    yforecasts.quarter+=yforecasts.ds.dt.quarter.astype(str)
    
    # 添加月份数据
    yforecasts=yforecasts.assign(month=lambda x:'M')
    yforecasts.month+=yforecasts.ds.dt.month.astype(str).str.zfill(2)
    # 添加星期数据
    yforecasts=yforecasts.assign(weekday=lambda x: x.ds.dt.weekday.replace({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}))
    
    #yforecasts=yforecasts.merge(hld[['ds','holiday']],how='left',on='ds')
    
    # 各成分比重
    tscomponents={} 
    
    quarter_ratio=yforecasts.groupby('quarter')['yearly'].mean().sort_index()
    month_ratio=yforecasts.groupby('month')['yearly'].mean().sort_index()
    weekday_ratio=yforecasts.groupby('weekday')['weekly'].mean()
    holiday_ratio=yforecasts.groupby('holiday')['holidays'].mean()
    tscomponents['yearly']={'quarter_ratio':quarter_ratio.to_dict()\
                            ,'month_ratio':month_ratio.to_dict()\
                            ,'quarter_max':list(quarter_ratio.sort_values().index[-1:])\
                            ,'month_max':list(month_ratio.sort_values().index[-2:])\
                            ,'month_min':list(month_ratio.sort_values().index[:2])}
    
    tscomponents['weekly']={'weekday_ratio':weekday_ratio.to_dict()\
                            ,'weekday_max':list(weekday_ratio.sort_values().index[-1:])\
                            ,'weekday_min':list(weekday_ratio.sort_values().index[:1])}
    
    tscomponents['holidays']={'holiday_ratio':holiday_ratio.to_dict()\
                              ,'holiday_max':list(holiday_ratio.sort_values().index[-1:])\
                              ,'holiday_min':list(holiday_ratio.sort_values().index[:1])}

    # 计算异常日期 anomaly_date
    w=forecasts[['ds','residual','bahld']]
    w1=w.loc[np.abs(w['residual']-w['residual'].mean())>=3*w['residual'].std(),:]
    if len(w1)>0:
        anomaly_date=list(w1.loc[((w1['bahld']>'BH14')&(w1['bahld']<'BH99'))|((w1['bahld']>'AH14')&(w1['bahld']<'AH99')),:]['ds'])
    else:
        anomaly_date=[]
    tscomponents['anomaly_date']=anomaly_date


    return tsdecompose,tscomponents
                       
 
Forecasts=pd.DataFrame()

params_best={}
columns=['cityname','mape','rmse','seasonality_prior_scale','holidays_prior_scale','changepoint_prior_scale','interval_width']
columns+=['seasonal','yearly','weekly','holidays','residual']
scores=pd.DataFrame(index=df['cityid'].unique(),columns=columns)
city_decompose=[]                       
for ind in dim_city.index:
    city=dim_city.loc[ind,'cityid']
    cityname=dim_city.loc[ind,'cityname']
    print('================ begin run : {} {}======================'.format(city,cityname))
    if 'cityid' in Forecasts and city in Forecasts['cityid'].unique():
        continue

    df_city=df.loc[df['cityid']==city,['ds','y']]
    cv_param={'horizon':30,'period':120,'initial':1095}
    param,best_scores,scores_tmp=ts_grid_search(df_city,holidays,cv_param=cv_param)
    print('best scores mape :{:.2f}%'.format(best_scores*100))
    params_best[city]=param
    scores.loc[city,['cityname','mape','rmse']]=[cityname,scores_tmp['mape'].min(),scores_tmp['rmse'].min()]
    
    scores.loc[city,['seasonality_prior_scale','holidays_prior_scale','changepoint_prior_scale','interval_width']]\
    =[param['seasonality_prior_scale'],param['holidays_prior_scale'],param['changepoint_prior_scale'],param['interval_width']]
    
    m=Prophet(holidays=holidays,**param)
    m.fit(df_city)
    forecasts=m.predict(df_city[['ds']])
    forecasts=forecasts.merge(df_city[['ds','y']],how='left',on='ds')
    forecasts=forecasts.assign(residual = lambda x:x.y-x.yhat)
    forecasts=forecasts.merge(hld[['ds','holiday','bahld']],how='left',on='ds') 
    
    
    tsdecompose,tscomponents=ts_decompose(forecasts)
    scores.loc[city,['seasonal','yearly','weekly','holidays','residual']]\
    =[tsdecompose['seasonal'],tsdecompose['yearly'],tsdecompose['weekly'],tsdecompose['holidays'],tsdecompose['residual']]
    tscomponents_tmp={'cityid':city,'cityname':cityname}
    tscomponents_tmp.update(tscomponents['yearly'])
    tscomponents_tmp.update(tscomponents['weekly'])
    tscomponents_tmp.update(tscomponents['holidays'])
    city_decompose.append(tscomponents_tmp)
    
    tmp=forecasts[['ds','y','yhat','trend','seasonal','yearly','weekly','holidays','residual']]
    tmp.insert(1,'cityid',city)
    tmp.insert(2,'cityname',cityname)
    Forecasts=pd.concat([Forecasts,tmp],axis=0)
   


city_decompose=pd.DataFrame(city_decompose)

scores.to_excel('scores.xlsx',index=True)
params_best=pd.DataFrame(params_best).T

                     

 