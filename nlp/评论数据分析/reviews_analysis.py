# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 21:33:06 2017

@author: gason
"""

import numpy as np
import pandas as pd
import re
import random

from gensim import corpora, models, similarities
import gensim
import jieba
import jieba.analyse as analyse

def clean_text(text):
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r'\s',"",text) #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"&hellip;|&mdash;|&ldquo;|&rdquo;", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    return text


#df = pd.read_csv("./data/jdcomments.csv")
df=pd.read_excel('./data/红米4A.xlsx')
df['creationTime']=pd.to_datetime(df['creationTime'])
df['creationTime']=df['creationTime'].map(lambda x:x.date())
df=df[df['content'].map(lambda x:isinstance(x,str))]
df['content']=df['content'].map(lambda x:clean_text(x))
df['contentLen']=df['content'].map(lambda x:len(x))

df=df[df['contentLen']>2]
df=df.drop_duplicates('content')


# 处理乱七八糟的评论

def rubbish_contents(s):
    if not(isinstance(s,str)):
        return True
    s=re.sub(r'\s',"",s)
    rubbish=[
    '吾',
    '女娲造人',
    '斧下去',
    '混沌初开',
    '神州平地一声雷',
    '人之初也',
    '实乃国之幸也'
    '只见顶天立地一金甲天神立于天地间',
    '天佑我大中华',
    '天崩地裂',
    '永不变心',
    '出淤泥之清莲',
    '辗转反侧无法忘怀',
    '无不让人感激涕零',
    '遂沐浴更衣',
    '以至茶饭不思',
    '我内心的那种激动才逐渐平静下来',
    '寝食难安',
    '屋内升起七彩祥云',
    '自觉七经八脉为之一畅',
    '焚香祷告后与人共赏此宝',
    '如果将来我再也遇不到了',
    '东哥之热心',
    '人皆赞叹不已',
    '凑齐银两',
    '此宝乃是天上物',
    '呜呼哀哉',
    '产品介绍果然句句实言',
    '本人对此卖家之仰慕如滔滔江水连绵不绝',
    '海枯石烂',
    '直到我毫不犹豫地把卖家的店收藏了',
    '这位英雄手持双斧',
    '我要以此评价奉献给世人赏阅',
    '阅商无数',
    '老板你实在是太好了',
    '不由得精神为之一振',
    '如果将来我再也买不到了']
    flag=False
    for rr in rubbish:
        if rr in s:
            flag=True
            continue
    tmp=0
    for t in s:
        if t==s[0]:
            tmp+=1
    if (tmp==len(s)) or (len(s)>5 and len(s)-tmp<=2):
        flag=True
        
    return flag
 
df=df[df['content'].map(lambda x:not(rubbish_contents(x)))]
   
df['keys']=df['content'].map(lambda x:','.join(analyse.extract_tags(x,topK=10,withWeight=False,allowPOS=())))
#df.to_excel('红米4A_处理后.xlsx')
#df0=df.copy()





from PIL import Image
import numpy as np
from wordcloud import WordCloud

comments_keys=pd.DataFrame(columns=['type','key','count','font_size', 'position_x','position_y'])

d = 'mask_circle.png'
mask = np.array(Image.open(d))
background_color='white'
# 好评数据
contents=' '.join(df.loc[df['score']==5,'content'])
wordcloud = WordCloud(background_color = background_color,font_path='DroidSansFallback.ttf', mask = mask).generate(contents)
keywords=[(w[0][0],w[0][1],w[1],w[2][0],w[2][1]) for w in wordcloud.layout_]
tmp=pd.DataFrame(keywords,columns=['key','count','font_size', 'position_x','position_y'])
tmp['type']='好评'
comments_keys=comments_keys.append(tmp)
wordcloud.to_image().save('好评词云.png')

# 中评数据
contents=' '.join(df.loc[(df['score']==3)|(df['score']==4),'content'])
wordcloud = WordCloud(background_color = background_color,font_path='DroidSansFallback.ttf', mask = mask).generate(contents)
keywords=[(w[0][0],w[0][1],w[1],w[2][0],w[2][1]) for w in wordcloud.layout_]
tmp=pd.DataFrame(keywords,columns=['key','count','font_size', 'position_x','position_y'])
tmp['type']='中评'
comments_keys=comments_keys.append(tmp)
wordcloud.to_image().save('中评词云.png')

# 中评数据
contents=' '.join(df.loc[df['score']<3,'content'])
wordcloud = WordCloud(background_color = background_color,font_path='DroidSansFallback.ttf', mask = mask).generate(contents)
keywords=[(w[0][0],w[0][1],w[1],w[2][0],w[2][1]) for w in wordcloud.layout_]
tmp=pd.DataFrame(keywords,columns=['key','count','font_size', 'position_x','position_y'])
tmp['type']='差评'
comments_keys=comments_keys.append(tmp)
wordcloud.to_image().save('差评词云.png')

comments_keys=pd.DataFrame(comments_keys,columns=['type','key','count','font_size', 'position_x','position_y'])
comments_keys.to_excel('评论关键词词云.xlsx',index=False)
















    
docs=df['content']
doclist = docs.values

f=open('.\\stopwords\\chinese.txt',encoding='utf-8')
stopwords=f.readlines()
stopwords=[s.strip() for s in stopwords]


texts = [[word for word in list(jieba.cut(doc)) if word not in stopwords] for doc in doclist]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

lda.print_topic(10, topn=5)

lda.print_topics(num_topics=20, num_words=5)            


































    
    