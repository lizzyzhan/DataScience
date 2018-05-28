# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:46:15 2018

@author: gason
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
sns.set()


'''
一些规范
df 是时间序列的数据，ds,y 是至少的两列
model:
- .history : 训练的历史数据
- .copy(cutoff=None): 参数拷贝
- .fit: 必须的列:ds,y
- .predict: 输入ds等，至少返回ds,yhat
'''


def _add_freq(df,freq=None):
    ''' 给df的ds列添加freq
    '''
    idx=pd.DatetimeIndex(df['ds'])
    if freq is None:
        if idx.freq is None:
            freq=pd.infer_freq(idx)
        else:
            return df
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('no discernible frequency found to ds')
    df.loc[:,'ds']=idx
    return df


def _cutoffs(df, horizon, k, period):
    """Generate cutoff dates

    Parameters
    ----------
    df: pd.DataFrame with historical data
    horizon: pd.Timedelta.
        Forecast horizon
    k: Int number.
        The number of forecasts point.
    period: pd.Timedelta.
        Simulated Forecast will be done at every this period.

    Returns
    -------
    list of pd.Timestamp
    """
    # Last cutoff is 'latest date in data - horizon' date
    # 先计算出最后一个 cutoff 日期
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]

    for i in range(1, k):
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
            # Next cutoff point is 'last date before cutoff in data - horizon'
            closest_date = df[df['ds'] <= cutoff].max()['ds']
            cutoff = closest_date - horizon
        if cutoff < df['ds'].min():
            #logger.warning(
            #    'Not enough data for requested number of cutoffs! '
            #    'Using {}.'.format(i))
            break
        result.append(cutoff)

    # Sort lines in ascending order
    return reversed(result)


def simulated_historical_forecasts(model, horizon, k, period=None):
    """Simulated Historical Forecasts.

    Make forecasts from k historical cutoff points, working backwards from
    (end - horizon) with a spacing of period between each cutoff.

    Parameters
    ----------
    model: Time Series Forecast Model class object.
        Fitted Time Series Forecast Model model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    k: Int number of forecasts point.
    period: Optional string with pd.Timedelta compatible style. Simulated
        forecast will be done at every this period. If not provided,
        0.5 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    cutoffs = _cutoffs(df, horizon, k, period)
    # 利用append 只是加入了单个的dataframe,然后通过过reduce再真正合起来
    predicts = []
    for cutoff in cutoffs:
        # Generate new object with copying fitting options
        # 复制model的参数，同时确保changepoint等参数保持正确
        m = model.copy(cutoff)
        # Train model
        m.fit(df[df['ds'] <= cutoff])
        # Calculate yhat
        index_predicted = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
        # 一般只需要df中的ds这一列，但允许其他列参数在里面.
        yhat = m.predict(df[index_predicted])
        # Merge yhat(predicts), y(df, original data) and cutoff
        # 在Time Series Forecast Model中 yhat的列是：['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        predicts.append(pd.concat([
            yhat,
            df[index_predicted][['y']].reset_index(drop=True),
            pd.DataFrame({'cutoff': [cutoff] * len(yhat)})
        ], axis=1))

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return reduce(lambda x, y: x.append(y), predicts).reset_index(drop=True)


def cross_validation(model, horizon, period=None, initial=None,score='MAPE'):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points. Beginning from initial,
    makes cutoffs with a spacing of period up to (end - horizon).

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Time Series Forecast Model class object. Fitted  model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will begin here. If not provided, 3 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    te = model.history['ds'].max()
    ts = model.history['ds'].min()
    # 后期可自动化，根据ds的freq给定
    horizon = pd.Timedelta(horizon)
    # 默认隔 0.5的horizon 取一个cutoff 
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    # 默认用 3倍horizon 时间的数据来训练 
    initial = 3 * horizon if initial is None else pd.Timedelta(initial)
    # 最大的cutoff数量
    k = int(np.ceil(((te - horizon) - (ts + initial)) / period))
    if k < 1:
        raise ValueError(
            'Not enough data for specified horizon, period, and initial.')
    return simulated_historical_forecasts(model, horizon, k, period)

    
def ForecastReport(forecasts):
    '''
    forecasts: ds,y,yhat,cutoff,horizon 等
    '''
    #result=[]
    forecasts['mape']=np.abs(forecasts['yhat']-forecasts['y'])/forecasts['y']
    forecasts['rmse']=(forecasts['yhat']-forecasts['y'])**2
    if 'horizon' in forecasts.columns:
        result=forecasts.groupby('horizon').agg({'mape':'mean','rmse':lambda x:np.sqrt(np.mean(x))})
    else:
        result=pd.Series()
        result['rmse']=np.sqrt(np.mean(forecasts['rmse']))
        result['mape']=np.mean(forecasts['mape'])
    #fig,ax=plt.subplots()
    #ax.plot(forecasts['y'],forecasts['yhat'],'.')
    g=sns.lmplot('y','yhat',data=forecasts)
    g.set_titles('y~yhat')  
    
