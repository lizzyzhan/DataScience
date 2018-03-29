# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:14:17 2017

@author: gason
"""

import pandas as pd
import numpy as np
import CustomerReviews as cr
import reportgen as rpt

import os




reviews_filename='国美（GOME）U7'
data=pd.read_excel('.\\reviews\\{}.xlsx'.format(reviews_filename))



myppt=rpt.Report()
myppt.add_cover(title=reviews_filename+'评论典型意见分析')
myppt.add_slide(data=pd.DataFrame(data['userLevelName'].value_counts()*100/len(data)),title='用户级别')


comments=cr.Reviews(texts=data['content'],scores=data['score'])    
comments.replace('synonyms.txt')
# 分词。此处用的是结巴分词工具.
# 添加了手机领域的专有词、以及产品特点词语，比如磨砂黑、玫瑰金等
comments.segment(product_dict='mobile_dict.txt',\
                 stopwords='.\\stopwords\\chinese.txt',\
                 add_words=list(data['productColor'].dropna().unique()))

# 去除无效评论
initial_words=['京东豆','数数','经济','杂交','今生今世','红红火火',\
               '彰显','荣华富贵','仰慕','滔滔不绝','永不变心',\
               '海枯石烂','天崩地裂']
comments.drop_invalid(initial_words=initial_words,max_rate=0.6)
    


slides_data=[]
s=comments.describe()
slides_data.append({'title':'summary','data':'\n'.join(['{}: {}'.format(k,s[k]) for k in s.index])})
keywords=comments.get_keywords()
summary='评论的关键词为：'+'|'.join(keywords)
comments.genwordcloud(filename='wordcloud_all',imshow=False);
myppt.add_slide(data='wordcloud_all.png',title='评论的词云',summary=summary)

for k in ['好评','中评','差评']:
    # textrank 关键词
    keywords=comments.get_keywords(comments.scores==k)
    summary='{} 的关键词为：'.format(k)+'|'.join(keywords)
    # 生成词云
    tmp={'好评':'good','中评':'middle','差评':'bad'}
    filename='wordcloud_{}'.format(tmp[k])
    comments.genwordcloud(comments.scores==k,filename=filename,imshow=False);
    myppt.add_slide(data=filename+'.png',title=k+'摘要',summary=summary)

features=comments.get_product_features(min_support=0.005)
features_new=list(set(features)-set(['有点','物流','想象','速度快'])\
                  |set(['拍照','照相','内存','续航','全面屏','面容识别','人脸解锁','钥匙串']))
features_opinion,feature_corpus=comments.features_sentiments(features_new,method='score')
features_opinion=features_opinion.sort_values('mention_count',ascending=False)

tmp=pd.DataFrame(features_opinion['mention_count']/len(comments))
myppt.add_slide(data=tmp,title='特征提及率')

tmp=features_opinion['p_positive'].sort_values(ascending=False)
myppt.add_slide(data=tmp,title='特征好评率')

myppt.save(reviews_filename+'评论典型意见分析.pptx')













