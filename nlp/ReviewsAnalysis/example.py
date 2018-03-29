# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:26:43 2017

@author: gason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CustomerReviews import Reviews

data=pd.read_csv('.\\data\\vivo_x20.csv')
color=list(data['productColor'].dropna().unique())

comments=Reviews(data['content'],data['score'],data['creationTime'])
comments.describe()


# 解决一词多义问题以及统一产品特征名词。比如触摸屏-->触屏等
comments.replace('synonyms.txt')
# 分词。此处用的是结巴分词工具，添加了手机领域的专有词、以及产品特点词语，比如磨砂黑、玫瑰金
comments.segment(product_dict='mobile_dict.txt',stopwords='.\\stopwords\\chinese.txt',add_words=color)
# 去除无效评论
initial_words=['经济','杂交','今生今世','红红火火','彰显','荣华富贵','仰慕','滔滔不绝','永不变心','海枯石烂','天崩地裂']
comments.drop_invalid(initial_words=initial_words,max_rate=0.6)
comments.describe()

'''
from sklearn import metrics
ss=comments.sentiments(method='snownlp')
ss1=pd.cut(ss,[-0.1,0.0139,0.0315,1],labels=['差评','中评','好评'])
metrics.accuracy_score()
metrics.roc_auc_score()
'''


for k in ['好评','中评','差评']:
    keywords=comments.get_keywords(comments.scores==k)
    print('{} 的关键词为：'.format(k)+'|'.join(keywords))
    comments.genwordcloud(comments.scores==k,filename='wordcloud of {}'.format(k));
    
    








