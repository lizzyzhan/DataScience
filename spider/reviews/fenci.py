# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:36:03 2017

@author: gason
"""

# 分词
import pandas as pd
from textblob import TextBlob

d=pd.read_csv('.\\data\\B007A2K9YS.csv')
s=d['review'][0]
model=TextBlob(s)

# stopwords

f=open('.\\stopwords\\german.txt',encoding='utf-8')
stopwords=f.readlines()
stopwords=[s.strip() for s in stopwords]

model=model.lower()
wc=model.word_counts

for k in wc:
    if k in stopwords:
        wc.pop(k)
