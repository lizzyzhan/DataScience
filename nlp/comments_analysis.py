# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:17:06 2017
@author: 10206913
"""

import numpy as np
import pandas as pd
import re
import os


from gensim import corpora,models,similarities
from gensim.models.ldamodel import LdaModel
import jieba
import jieba.analyse as analyse





import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.univariate_selection import chi2, f_classif


#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
    text = re.sub("iphone", "苹果", text) #英文转中文
    text = re.sub("sumsung", "三星", text) #英文转中文
    text = re.sub("xiaomi", "小米", text) #英文转中文
    text = re.sub("hongmi", "红米", text) #英文转中文
    text = re.sub("honor", "荣耀", text) #英文转中文
    text = re.sub("huawei", "华为", text) #英文转中文
    text = re.sub("zte", "中兴", text) #英文转中文
    text = re.sub("nubia", "努比亚", text) #英文转中文
    text = '' if re.findall('^[a-z\d\.\s]+$',text) else text # 全是字母/数字/空格/.就去掉
    text = re.sub('[a-z\.]{6,}',' ',text)#去掉英文单词，无效英文等。仅保留产品型号等英文字母
    text = re.sub(u'[^\u4e00-\u9fa5\u0000-\u007a]+',' ',text)#只保留中文、以及Unicode到字母z的那段（不包含中文标点符号）
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
    texts: pd.Series格式，将分词后的结果用空格隔开，如： 这 手机 不错
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
    if isinstance(texts,pd.core.series.Series):
        index=texts.index
    else:
        index=range(len(texts))
    def _jieba_cut(doc):
        s=' '.join([word for word in list(jieba.cut(doc)) if word not in stopwords])
        # 去掉分词结果中全是字母或数字的
        s=re.sub('\s[a-z\.]+\s|^[a-z\.]+\s|\s[a-z\.]+$|\s[\d\.]+\s|^[\d\.]+\s|\s[\d\.]+$',' ',s)
        s=s.strip()
        s=re.sub(r'\s+',' ',s)
        return s
    texts=map(_jieba_cut,texts)
    texts=pd.Series(texts,index=index)
    return texts



def polysemy_replace(texts):
    '''
    处理分词结果中的一词多义
    
    '''
    # 持续改善
    polysemy={'老妈':'妈妈',
              '老爸':'爸爸',
              '老年人':'老人',
              '触摸屏':'触屏',
              '老人家':'老人',
              '[哈]+':'哈哈',
               '[好]+':'好',
                '还可以':'还行'}
    def _polysemy_replace(text):
        for k in polysemy:
            text=re.sub(k,polysemy[k],text)
        return text
    texts=pd.Series(map(_polysemy_replace,texts))
    return texts



def _isrubbish(x,keywords):
    flag=False
    x=('%s'%x).strip()
    x=re.sub(r'\s+',' ',x)
    if len(x)==0:
        flag=True
    else:
        words=x.split(' ')
        rate=len(set(words)&set(keywords))/len(words)
        flag=True if rate>=0.5 else False
    return flag


def cleaning(texts,initial_words=[]):
    texts_vec,words=text2vec(texts)
    #words=list(words)
    texts_feature=np.dot(texts_vec.T,texts_vec)
    feature_norm=np.sqrt(texts_feature.diagonal())
    texts_feature=texts_feature/np.dot(feature_norm.reshape((-1,1)),feature_norm.reshape(1,-1))
    texts_feature=texts_feature-np.eye(texts_feature.shape[0])  
    similar_words=[]
    for w in initial_words:
        if w in words:
            ind=np.argwhere(words==w)[0][0]
            #ind=words.index(w)
            tmp=texts_feature[:,ind]
            a,b=np.where(tmp>=0.8)
            similar_words+=[words[i] for i in a]
    invalid_texts=texts.map(lambda x:_isrubbish(x,similar_words))        
    return invalid_texts

 

def text2vec(texts,vec_model='idf'):
    '''
    向量化
    '''
    if vec_model == "idf":
        vectorizer = TfidfVectorizer(min_df=2,max_df=0.95,ngram_range=(1,2),sublinear_tf=True)
    else:
        vectorizer = CountVectorizer()
    texts_vec = vectorizer.fit_transform(texts)
    #texts_vec=texts_vec.toarray()
    words=np.array(vectorizer.get_feature_names())
    return texts_vec,words


def feature_engineering(texts_vec,scores,words=None,n_features=1500,score_func=None):
    '''特征工程
    基于评论的评分筛选出特定数目的特征
    '''
    n_features_initial=texts_vec.shape[1]
    if words is None:
        words=np.arange(n_features_initial)
    n_features=min(n_features_initial,n_features)
    if score_func is None:
        score_func=f_classif
    selector = SelectKBest(score_func,k=n_features).fit(texts_vec,scores)
    informative_words_index = selector.get_support(indices=True)
    words = np.array([words[i] for i in informative_words_index])
    texts_vec=texts_vec[:,informative_words_index]
    return texts_vec,words

    
def comments_wordcloud(contents,filename='词云.png'):
    from PIL import Image
    from wordcloud import WordCloud
    d = 'mask_circle.png'
    mask = np.array(Image.open(d))
    background_color='white'
    # 好评数据
    if not(isinstance(contents,str)):
        contents=' '.join(contents)
    wordcloud = WordCloud(background_color = background_color,font_path='DroidSansFallback.ttf', mask = mask)
    wordcloud.generate(contents)
    wordcloud.to_image().save(filename)
    keywords=[(w[0][0],w[0][1],w[1],w[2][0],w[2][1]) for w in wordcloud.layout_]
    comments_keys=pd.DataFrame(keywords,columns=['key','count','font_size', 'position_x','position_y'])
    return comments_keys



def find_topic(texts_vec, words,topic_model='lda',n_topics=10,n_words=10,thr=0.01,**kwargs):
    """Return a list of topics from texts by topic models - for demostration of simple data
    texts: array-like strings
    topic_model: {"nmf", "svd", "lda", "kmeans"} for LSA_NMF, LSA_SVD, LDA, KMEANS (not actually a topic model)
    n_topics: # of topics in texts
    vec_model: {"tf", "tfidf"} for term_freq, term_freq_inverse_doc_freq
    thr: threshold for finding keywords in a topic model
    """
    ## 1. topic finding
    topic_models = {"nmf": NMF, "svd": TruncatedSVD, "lda": LatentDirichletAllocation, "kmeans": KMeans}
    topicfinder = topic_models[topic_model](n_topics, **kwargs).fit(texts_vec)
    topic_dists = topicfinder.components_ if topic_model is not "kmeans" else topicfinder.cluster_centers_
    #topic_dists /= topic_dists.sum(axis = 1).reshape((-1, 1))# np.array, n_topics*n_features
    ## 2. keywords for topics
    ## Unlike other models, LSA_SVD will generate both positive and negative values in topic_word distribution,
    ## which makes it more ambiguous to choose keywords for topics. The sign of the weights are kept with the
    ## words for a demostration here

    def _topic_keywords_thr(topic_dist):
        '''根据值得大小选择
        '''
        # 根据关键词的数目选择
        topic_dist /= topic_dist.max()
        keywords_index = np.abs(topic_dist) >= thr
        #keywords_prefix = np.where(np.sign(topic_dist) > 0, "", "^")[keywords_index]
        keywords=' | '.join(words[keywords_index])
        #keywords = " | ".join(map(lambda x: "".join(x), zip(keywords_prefix, words[keywords_index])))
        return keywords

    def _topic_keywords_n(topic_dist):
        '''根据关键词数目选择
        '''
        # 根据关键词的数目选择
        topic_dist /= topic_dist.sum()
        ind=np.argsort(topic_dist)[-1:-1*n_words-1:-1]
        keywords=' | '.join([words[ii] for ii in ind])
        return keywords
    if n_words is not None:
        topic_keywords = map(_topic_keywords_n, topic_dists)
    else:
        topic_keywords = map(_topic_keywords_thr, topic_dists)
    return "\n\n".join("Topic %i: %s" % (i, t) for i, t in enumerate(topic_keywords))



def gensim_lda(texts,n_topics=10,n_words=10,vec_model='tf'):
    dictionary = corpora.Dictionary([doc.split(' ') for doc in texts])
    corpus = [dictionary.doc2bow(text.split(' ')) for text in texts]
    if vec_model in ['idf','tfidf']:
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)
    topics_keywords=lda.show_topics(num_topics=n_topics, num_words=n_words,formatted=False)
    print('\n'.join(['主题 {}: {}'.format(i,' | '.join([k[0] for k in topic[1]])) for i,topic in enumerate(topics_keywords)]))
    return topics_keywords



data=load_data('./data/红米4A.xlsx')
texts=jieba_cut(data['content'],'mobile_dict.txt','.\\stopwords\\chinese.txt')
texts=polysemy_replace(texts)
scores=data['score'].reset_index(drop=True)

# 去除垃圾评论 
initial_words=['女娲造人','混沌初开','天崩地裂','永不变心','寝食难安','七彩祥云',\
'七经八脉', '焚香祷告','凑齐银两','呜呼哀哉','海枯石烂','阅商无数','顶天立地','欣喜若狂',\
'乃是','天上','三斧','小生', '七彩', '祥云','吾','宝物']
invalid=cleaning(texts,initial_words)
texts=texts[~invalid]
scores=scores[~invalid]


N=len(texts)

print('-'*20+'【好评】'+'-'*20)
texts_tmp=texts[scores==5]
print('一共有{}条，占比 {:.1f}%'.format(len(texts_tmp),len(texts_tmp)*100/N))
w=comments_wordcloud(texts_tmp,filename='好评词云.png');
gensim_lda(texts_tmp,n_topics=10,n_words=10);
          
print('-'*20+'【中评】'+'-'*20)
texts_tmp=texts[(scores<5)&(scores>=3)]
print('一共有{}条，占比 {:.1f}%'.format(len(texts_tmp),len(texts_tmp)*100/N))
w=comments_wordcloud(texts_tmp,filename='中评词云.png');
gensim_lda(texts_tmp,n_topics=10,n_words=10);
          
print('-'*20+'【差评】'+'-'*20)
texts_tmp=texts[scores<5]
print('一共有{}条，占比 {:.1f}%'.format(len(texts_tmp),len(texts_tmp)*100/N))
w=comments_wordcloud(texts_tmp,filename='差评词云.png');
gensim_lda(texts_tmp,n_topics=10,n_words=10);

         
