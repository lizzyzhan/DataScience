# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:42:50 2017

@author: 10206913
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 准备迁移到config中去
'''
字段：别名，数据类型，优先输出类型
'''
chart_type_list={\
"COLUMN_CLUSTERED":['柱状图','ChartData','pptx'],\
"BAR_CLUSTERED":['条形图','ChartData','pptx'],
'HIST':['分布图,KDE','XChartData','matplotlib']}
chart_type_list=pd.DataFrame(chart_type_list)


fontlist=['calibri.ttf','simfang.ttf','simkai.ttf','simhei.ttf','simsun.ttc','msyh.ttf','msyh.ttc','MSYH.TTC']
find_paths=['C:\\Windows\\Fonts','']
# fontlist 越靠后越优先，findpath越靠后越优先
for find_path in find_paths:
    for f in fontlist:
        if os.path.exists(os.path.join(find_path,f)):
            font_path=os.path.join(find_path,f)

    fig, ax = plt.subplots()
    #ax.grid('on')






def plot(data,figure_type='auto',chart_type='auto'):
    '''auto choose the best chart type to draw the data
    paremeter
    -----------
    figure_type: 'mpl' or 'pptx' or 'html'
    chart_type: 'hist' or 'dist' or 'kde' or 'bar' ......
    
    return
    -------
    chart:dict format.
    .type: equal to figure_type
    .fig: only return if type == 'mpl'
    .ax:
    .chart_data:
    .chart_type:
    
    '''
    
    # 判别部分
    
    # 绘制部分
    chart={}   
    if figure_type in ['mpl','matplotlib']:
        chart['type']='mpl'
        fig,ax=plt.subplots()
        if chart_type=='HIST':
            for c in data.columns:
                sns.kdeplot(data[c],shade=True,ax=ax)
            ax.legend()
        elif chart_type == 'SCATTER':
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.axhline(y=0, linestyle='-', linewidth=1.2, alpha=0.6)
            ax.axvline(x=0, linestyle='-', linewidth=1.2, alpha=0.6)
            color=['blue','red','green','dark']
            if not isinstance(data,list):
                data=[data]
            for i,dd in enumerate(data):     
                ax.scatter(dd.iloc[:,0], dd.iloc[:,1], c=color[i], s=50,
                           label=dd.columns[1])
                for _, row in dd.iterrows():
                    ax.annotate(row.name, (row.iloc[0], row.iloc[1]), color=color[i],fontproperties=myfont,fontsize=10)
            ax.axis('equal')
            ax.legend()
            
        chart['fig']=fig
        chart['ax']=ax
        return chart




           
        
