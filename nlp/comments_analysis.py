# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:17:06 2017

@author: 10206913
"""

import numpy as np
import pandas as pd
import re
import os


from gensim import corpora, models, similarities
import gensim
import jieba
import jieba.analyse as analyse





import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans



def clean_text(text):
    text='%s'%text
    text=text.lower()# 小写
    text = text.replace('\r\n'," ") #新行，我们是不需要的
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r'\s',"",text) #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"&hellip;|&mdash;|&ldquo;|&rdquo;", "", text) #网页符号，没意义
    text = re.sub(r"&[a-z]{3,10};", " ", text) #网页符号，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    text = re.sub("apple", "苹果", text) #英文转中文
    text = re.sub("sumsung", "三星", text) #英文转中文
    text = re.sub("xiaomi", "小米", text) #英文转中文
    text = re.sub("hongmi", "红米", text) #英文转中文
    text = re.sub("honor", "荣耀", text) #英文转中文
    text = re.sub("huawei", "华为", text) #英文转中文
    text = re.sub("zte", "中兴", text) #英文转中文
    text = re.sub("nubia", "努比亚", text) #英文转中文
    text = '' if re.findall('^[a-z\d\.\s]+$',text) else text # 全是字母/数字/空格/.就去掉
    return text



def load_data(filename):
    '''
    评论那一列名字为 content
    该函数对数据作粗处理，包含以下几个方面
    1、删除文本中无用信息，如邮件地址、时间、网址等等
    2、删除评论字数少于3个字符的评论
    '''
    savetype=os.path.splitext(filename)[1][1:]
    if (savetype==u'xlsx') or (savetype==u'xls'):
        df=pd.read_excel(filename)
    elif savetype==u'csv':
        df=pd.read_csv(filename)
    else:
        print('con not read file!')
        return None
    df['content']=df['content'].map(lambda x:clean_text(x))
    texts=df.loc[df['content'].map(lambda x:len('%s'%x))>2,:]
    return texts


def jieba_cut(texts,add_words=[],stopwords=[]):
    '''
    对中文文本进行分词，并将分词结果用空格隔开
    注：该函数不会删除样本
    parameter
    ---------
    texts: 可迭代文本对象，每一个对应着一个一份文档
    add_words: 自己添加的 jiaba 词典
    stopwords：停止词，用于分词
    
    return
    ------
    texts: 列表格式，将分词后的结果用空格隔开，如： 这 手机 不错
    '''
    
    if isinstance(add_words,str):
        f=open(add_words,encoding='utf-8')
        add_words=f.readlines()
        add_words=[s.strip() for s in add_words]       

    texts_tmp=','.join(texts)
    # 手机内存、屏幕尺寸等
    add_words+=list(set(re.findall('[\d\.\+]{1,5}[g寸万]{1}',texts_tmp)))
    # 识别品牌名称+型号，如红米4a，骁龙835
    brand_name=['小米','红米','华为','荣耀','mate','三星','苹果','oppo','vivo',\
    '魅族','锤子','坚果','美图手机','美图','联想','高通骁龙','骁龙','联发科']
    for b in brand_name:
        add_words+=list(set(re.findall(b+'[\da-z]*',texts_tmp)))
    for word in add_words:
        jieba.add_word(word)
    if isinstance(stopwords,str):
        f=open(stopwords,encoding='utf-8')
        stopwords=f.readlines()
        stopwords=[s.strip() for s in stopwords]
    texts = [' '.join([word for word in list(jieba.cut(doc)) if word not in stopwords]) for doc in texts]
    texts=[re.sub('\s[a-z\.]+\s|^[a-z\.]+\s|\s[a-z\.]+$|\s[\d\.]+\s|^[\d\.]+\s|\s[\d\.]+$',' ',s) for s in texts]# 去掉分词结果中全是字母或数字的
    return texts



def polysemy_replace(texts,words=[]):
    '''
    处理分词结果中的一词多义
    
    '''
    # 持续改善
    return texts


def cleaning(texts,initial_words=[]):
    '''
    清除垃圾评论
    '''
    vectorizer = TfidfVectorizer()
    text_vec = vectorizer.fit_transform(texts)

    

def text2vec(texts,vec_model='tfidf'):
    '''
    向量化
    '''
    vectorizer = CountVectorizer() if vec_model == "tf" else TfidfVectorizer()
    texts_vec = vectorizer.fit_transform(texts)
    #texts_vec=texts_vec.toarray()
    words=vectorizer.get_feature_names()
    return texts_vec,words
    





data=load_data('./data/红米4A.xlsx')
texts=jieba_cut(data['content'],'mobile_dict.txt','.\\stopwords\\chinese.txt')

texts_vec,words=text2vec(texts)

texts_feature=np.dot(texts_vec.T,texts_vec)

for w in initial_words:
    if w in words:
        ind=words.index(w)
        texts_feature[:,ind].sort()




