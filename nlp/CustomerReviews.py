# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:17:06 2017
@author: JSong
"""

import numpy as np
import pandas as pd
import re
import os


from gensim import corpora,models,similarities
from gensim.models.ldamodel import LdaModel
import jieba
import jieba.analyse as analyse
from orangecontrib import associate
# associate.frequent_itemsets(X, min_support=0.2)
# associate.association_rules(itemsets, min_confidence, itemset=None)
from snownlp import SnowNLP



import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.univariate_selection import chi2, f_classif


#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Reviews():
    """初始化"""
    def __init__(self, contents,scores=None,data=None,language=None):
        '''初始化需要两个数据，一份是评论，一份是评分
        如果给定data,则contents和scores是对应的列名字，否则contents是具体的数组
        '''
        if data is not None:
            texts=data[contents]
            scores=data[scores]
        elif scores is None:
            texts=contents['content']
            scores=contents['score']
        else:
            texts=pd.Series(contents)
            scores=pd.Series(scores)
        texts=texts.map(lambda x:clean_text(x))
        ind=texts.map(lambda x:len('%s'%x)>2)
        texts=texts[ind]
        scores=scores[ind]
        texts=texts.reset_index(drop=True)
        scores=scores.reset_index(drop=True)
        self.texts_raw=texts#原始语料
        if language in ['english','en']:
            # 部分语系不需要分词
            self.texts_seg=self.texts_raw
        else:
            self.texts_seg=None
        self.texts_vec_idf=None#(向量化稀疏数组,words)
        self.texts_vec_tf=None#(向量化稀疏数组,words)
        self.scores=scores
        self.pos={}

    def find_keywords(self,keywords):
        '''根据给定的关键词在评论预料中查找到相应的句子（非整条评论）
        keywords='拍照|照片'
        暂时只适合中文
        '''
        def _keywords_find(s):
            kf=re.compile('[\u4e00-\u9fa5\u0061-\u007a\u0030-\u0039]+'+keywords+'[\u4e00-\u9fa5\u0061-\u007a\u0030-\u0039]+')
            tmp=re.findall(kf,s)
            if tmp:
                sentenses=' | '.join(tmp)
            else:
                sentenses=np.nan
            return sentenses
        texts_keywords=self.texts_raw.map(_keywords_find)
        texts_keywords=texts_keywords[texts_keywords.notnull()]
        return texts_keywords


    
    def drop_invalid(self,initial_words=[]):
        '''去除无效评论
        1、根据给定的初始无效关键词，寻找所有相似的词语
        2、统计无效关键词的占比，大于80%的则判定为无效评论
        parameter
        --------
        texts: 词语用空格隔开的语料
        
        return
        ------
        返回的是逻辑索引
        '''
        if self.texts_seg is not None:
            print('该对象还未分词，请检查')
            return self
       
        texts_vec,words=text2vec(self.texts_seg)
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
        invalid=self.texts_seg.map(lambda x:_isrubbish(x,similar_words))
        self.texts_raw=self.texts_raw[~invalid]
        if id(self.texts_raw) != id(self.texts_seg):
            self.texts_seg=self.texts_seg[~invalid]
        self.scores=self.scores[~invalid]
        return self


    def segment(self,product_dict=[],stopwords=[]):
        '''
        对中文文本进行分词，并将分词结果用空格隔开
        parameter
        ---------
        product_dict: 自己添加的 jiaba 词典
        stopwords：停止词，用于分词
        
        具体参见函数：jieba_cut
        '''
        texts_seg,words_pos=jieba_cut(self.texts_raw,add_words=product_dict,stopwords=stopwords,POS=True)
        self.texts_seg=texts_seg
        self.pos=words_pos
        return self


    def vectorizer(self,vec_model='idf',store=True):
        '''对语料向量化
        vec_model: tfidf/idf、tf、word2vec等
        对象默认会存储idf和tf分词结果，便于复用
        '''       
        if vec_model in ['idf','tfidf']:
            myvectorizer = TfidfVectorizer(min_df=2,max_df=0.95,ngram_range=(1,2),sublinear_tf=True)
        else:
            myvectorizer = CountVectorizer()
        texts_vec = myvectorizer.fit_transform(self.texts_seg)
        #texts_vec=texts_vec.toarray()
        words=np.array(myvectorizer.get_feature_names())
        if store and (vec_model in ['idf','tfidf']):
            self.texts_vec_idf=(texts_vec,words)
        elif store and (vec_model in ['tf']):
            self.texts_vec_tf=(texts_vec,words)
        return texts_vec,words
     


    def get_product_features(self,min_support=0.005):
        '''从产品评论中挖掘出产品特征词
        parameter
        --------
        texts: 空格分开的语料
        pos：词性标注dict，关注其中的名词：n和形容词：a
        min_support: 最小支持度
        
        return
        ------
        features:特征词列表
        '''
    
        if self.texts_vec_tf is None:
            texts_vec_tf,words=text2vec(self.texts_seg,vec_model='tf')
        else:
            texts_vec_tf,words=self.texts_vec_tf
        # 第一步，利用关联分析找到初始的特征词
        gen=associate.frequent_itemsets(texts_vec_tf, min_support=min_support)
        sup1=[[words[i] for i in s[0]] for s in gen if len(s[0])==2]

        # 第二步 筛选出 一个为名词，一个为形容词的特征，同时将名词排前面
        sup2=[sorted(s,key=lambda x:self.pos[x]!='n') for s in sup1 if set([self.pos[s[0]],self.pos[s[1]]])==set(['n','a'])]
    
        # 第三步 从中选出名词和形容词配对那些特征词
        sup3=[]
        for kw in sup2:
            dis=dis_of_pairwords(kw,self.texts_seg)
            if dis<4:
                sup3.append(kw)     
        
        features_frequent=list(set([s[0] for s in sup3]))
        opinion_words=list(set([s[1] for s in sup3]))
        # 第四步 通过意见词找其他词
        def _feature_find(texts,opinion):
            # 计算由keywords_agg()生成的语料
            pattern1=re.compile(r'([^\s]*?)/n[^n]*?'+opinion)
            pattern2=re.compile(r''+opinion+'/a[^n]*?\s([^\s]*?)/n')
            feature_words=[]
            for docs in texts: 
                doc=docs.split(' | ')
                for s in doc:
                    words=jieba.posseg.cut(s)
                    tmp=' '.join([w.word+'/'+w.flag for w in words])
                    fw1=re.findall(pattern1,tmp)
                    fw2=re.findall(pattern2,tmp)
                    feature_words+=fw1
                    feature_words+=fw2
            return feature_words
            
            # 找到里面的名词再找到意见词，返回距离最近的那一组
        fw_raw=[]
        for ow in opinion_words:
            texts_key=self.find_keywords(ow)
            fw_raw+=_feature_find(texts_key,ow)
        features_infrequent=[fw for fw in set(fw_raw) if fw_raw.count(fw)>2]
        features=list(set(features_frequent+features_infrequent))
        return features


        
        

def text2vec(texts,vec_model='idf'):
    '''
    向量化
    '''
    if vec_model in ['idf','tfidf']:
        myvectorizer = TfidfVectorizer(min_df=2,max_df=0.95,ngram_range=(1,2),sublinear_tf=True)
    else:
        myvectorizer = CountVectorizer()
    texts_vec = myvectorizer.fit_transform(texts)
    #texts_vec=texts_vec.toarray()
    words=np.array(myvectorizer.get_feature_names())
    return texts_vec,words
        



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
    texts=texts.reset_index(drop=True)
    return texts


def clean_text(text):
    text='%s'%text
    text=text.lower()# 小写
    text = text.replace('\r\n'," ") #新行，我们是不需要的
    text = text.replace('\n'," ") #新行，我们是不需要的
    text = re.sub(r"-", " ", text) #把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text) #日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"&hellip;|&mdash;|&ldquo;|&rdquo;", "", text) #网页符号，没意义
    text = re.sub(r"&[a-z]{3,10};", " ", text) #网页符号，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    text = '' if re.findall('^[a-z\d\.\s]+$',text) else text # 全是字母/数字/空格/.就去掉
    text = re.sub('[a-z\.]{6,}',' ',text)#去掉英文单词，无效英文等。仅保留产品型号等英文字母
    text = re.sub('，',',',text)#
    text = re.sub('。','.',text)#
    text = re.sub('！','!',text)#
    text = re.sub('？','?',text)#
    text = re.sub(u'[^\u4e00-\u9fa5\u0000-\u007a]+',' ',text)#只保留中文、以及Unicode到字母z的那段（不包含中文标点符号）
    return text



def keywords_agg(texts,keywords):
    '''根据给定的关键词在评论预料中查找到相应的句子（非整条评论）
    keywords='拍照|照片'
    只适合中文
    '''
    def _keywords_find(s):
        kf=re.compile('[\u4e00-\u9fa5\u0061-\u007a\u0030-\u0039]+'+keywords+'[\u4e00-\u9fa5\u0061-\u007a\u0030-\u0039]+')
        tmp=re.findall(kf,s)
        if tmp:
            sentenses=' | '.join(tmp)
        else:
            sentenses=np.nan
        return sentenses
    texts_new=texts.map(_keywords_find)
    texts_new=texts_new[texts_new.notnull()]
    return texts_new


def jieba_cut(texts,add_words=[],stopwords=[],POS=False):
    '''
    对中文文本进行分词，并将分词结果用空格隔开
    注：该函数不会删除样本
    parameter
    ---------
    texts: 可迭代文本对象，每一个对应着一个一份文档
    add_words: 自己添加的 jiaba 词典
    stopwords：停止词，用于分词
    POS：词性标注，默认不标注
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
        s=' '.join([word for word in jieba.cut(doc) if word not in stopwords])
        # 去掉分词结果中全是字母或数字的
        s=re.sub('\s[a-z\.]+\s|^[a-z\.]+\s|\s[a-z\.]+$|\s[\d\.]+\s|^[\d\.]+\s|\s[\d\.]+$',' ',s)
        s=s.strip()
        s=re.sub(r'\s+',' ',s)
        #s=re.sub(u'[^\u4e00-\u9fa5\u0061-\u007a\u0030-\u0039\u0020]+','',s)#只保留中文、以及Unicode到字母z的那段（不包含中文标点符号）
        return s
    if POS:
        words_pos={}
        texts_new=[]
        for doc in texts:
            words=jieba.posseg.cut(doc)
            tmp=[(w.word,w.flag) for w in words if w.word not in stopwords]
            words_pos.update(tmp)
            s=' '.join([t[0] for t in tmp])
            # 去掉分词结果中全是字母或数字的
            s=re.sub('\s[a-z\.]+\s|^[a-z\.]+\s|\s[a-z\.]+$|\s[\d\.]+\s|^[\d\.]+\s|\s[\d\.]+$',' ',s)
            s=s.strip()
            s=re.sub(r'\s+',' ',s)
            texts_new.append(s)
        texts=pd.Series(texts_new,index=index)
        return texts,words_pos
    else:
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
                '还可以':'还行',
                'apple':'苹果',
                'iphone':'苹果',
                'sumsung':'三星',
                'xiaomi':'小米',
                'hongmi':'红米',
                'honor':'荣耀',
                'huawei':'华为',
                'zte':'中兴',
                'nubia':'努比亚'}
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



def text2vec(texts,vec_model='idf'):
    '''
    向量化
    '''
    if vec_model in ['idf','tfidf']:
        myvectorizer = TfidfVectorizer(min_df=2,max_df=0.95,ngram_range=(1,2),sublinear_tf=True)
    else:
        myvectorizer = CountVectorizer()
    texts_vec = myvectorizer.fit_transform(texts)
    #texts_vec=texts_vec.toarray()
    words=np.array(myvectorizer.get_feature_names())
    return texts_vec,words




def cleaning(texts,initial_words=[]):
    '''去除无效评论
    1、根据给定的初始无效关键词，寻找所有相似的词语
    2、统计无效关键词的占比，大于80%的则判定为无效评论
    parameter
    --------
    texts: 词语用空格隔开的语料
    
    return
    ------
    返回的是逻辑索引
    '''
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
    '''生成词云
    后期进一步优化该函数，模板mask需要自定义
    '''
    from PIL import Image
    from wordcloud import WordCloud
    d = 'mask_circle.png'
    # mask 是RGBA模式，最后一个分量是alpha通道
    mask = np.array(Image.open(d))
    background_color='white'
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
    '''主题模型，和上面那个函数，优先使用该函数
    parameter
    ---------
    texts: 每个词用空格隔开的语料，N*1
    n_topics: 主题数
    n_words: 每个主题输出的词语数
    vec_model: 向量化方法，默认是tf
    '''
    dictionary = corpora.Dictionary([doc.split(' ') for doc in texts])
    corpus = [dictionary.doc2bow(text.split(' ')) for text in texts]
    if vec_model in ['idf','tfidf']:
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)
    topics_keywords=lda.show_topics(num_topics=n_topics, num_words=n_words,formatted=False)
    print('\n'.join(['主题 {}: {}'.format(i,' | '.join([k[0] for k in topic[1]])) for i,topic in enumerate(topics_keywords)]))
    return topics_keywords


# =====================Opinion================================


def dis_of_pairwords(s,texts):
    '''返回两个词在评论预料中的距离
    parametre
    --------
    s:['手机','不错']
    texts: 语料
    '''
    dis=[]
    pattern=re.compile(s[0]+'.*?'+s[1]+'|'+s[1]+'.*?'+s[0])
    for text in texts:
        tmp=re.findall(pattern,text)
        dis0=len(tmp[0])-len(s[0])-len(s[1]) if tmp else np.nan
        dis.append(dis0)
    dis=pd.Series(dis)
    dis=dis[dis.notnull()].quantile(0.05,'nearest')
    return dis


def product_features(texts,pos,min_support=0.005):
    '''从产品评论中挖掘出产品特征词
    parameter
    --------
    texts: 空格分开的语料
    pos：词性标注dict，关注其中的名词：n和形容词：a
    min_support: 最小支持度
    
    return
    ------
    features:特征词列表
    '''

    texts_vec,words=text2vec(texts,vec_model='tf')
    gen=associate.frequent_itemsets(texts_vec, min_support=0.005)
    sup=list(gen)
    sup1=[[words[i] for i in s[0]] for s in sup if len(s[0])==2]
    # 筛选出特征
    # 筛选出 一个为名词，一个为形容词的特征，同时将名词排前面
    sup2=[sorted(s,key=lambda x:pos[x]!='n') for s in sup1 if set([pos[s[0]],pos[s[1]]])==set(['n','a'])]
    # 筛选出在句子中较为接近的词

    # 从中选出名词和形容词配对那些特征词
    sup3=[]
    for kw in sup2:
        dis=dis_of_pairwords(kw,texts)
        if dis<4:
            sup3.append(kw)     
    
    features_frequent=list(set([s[0] for s in sup3]))
    opinion_words=list(set([s[1] for s in sup3]))
    # 通过意见词找其他词
    def _feature_find(texts,opinion):
        # 计算由keywords_agg()生成的语料
        pattern1=re.compile(r'([^\s]*?)/n[^n]*?'+opinion)
        pattern2=re.compile(r''+opinion+'/a[^n]*?\s([^\s]*?)/n')
        feature_words=[]
        for docs in texts: 
            doc=docs.split(' | ')
            for s in doc:
                words=jieba.posseg.cut(s)
                tmp=' '.join([w.word+'/'+w.flag for w in words])
                fw1=re.findall(pattern1,tmp)
                fw2=re.findall(pattern2,tmp)
                feature_words+=fw1
                feature_words+=fw2
        return feature_words
        
        # 找到里面的名词再找到意见词，返回距离最近的那一组
    fw_raw=[]
    for ow in opinion_words:
        texts_key=keywords_agg(data['content'],ow)
        fw_raw+=_feature_find(texts_key,ow)
    features_infrequent=[fw for fw in set(fw_raw) if fw_raw.count(fw)>2]
    features=list(set(features_frequent+features_infrequent))
    return features



def features_sentiments(texts_raw,features,thr=0.5):
    '''产品特征词的情感分析
    parameter
    --------
    texts_raw: 中文原始语料，非中文代码需要重新修改过
    features: 产品特征词，如：内存、续航、拍照等
    thr：情感判断，当大于thr时，判定为正面意见。默认为0.5
    return
    ------
    features_opinion：DataFrame 格式，各个特征词的提及数、正面意见占比，负面意见占比
    features_corpus：Series 格式，原始语料中提及到特征的句子
    '''
    
    features_opinion=pd.DataFrame(index=features,columns=['total','mention_count','p_positive','p_negative'])
    features_corpus=pd.Series(index=features)
    N=len(texts_raw)
    for fw in features:
        texts_fw=keywords_agg(texts_raw,fw)
        features_corpus[fw]=' || '.join(texts_fw)
        sc=texts_fw.map(lambda x:SnowNLP(x).sentiments)
        p=len(sc[sc>thr])/len(sc) if len(sc)>0 else np.nan
        features_opinion.loc[fw,:]=[N,len(sc),p,1-p]   
        #features_opinion[fw]=(len(sc),'{:.2f}%'.format(p*100),'{:.2f}%'.format(100-p*100))
    return features_opinion,features_corpus


# ======================================================================
initial_words=['女娲造人','混沌初开','天崩地裂','永不变心','寝食难安','七彩祥云',\
'七经八脉', '焚香祷告','凑齐银两','呜呼哀哉','海枯石烂','阅商无数','顶天立地','欣喜若狂',\
'乃是','天上','三斧','小生', '七彩', '祥云','吾','宝物']

data=pd.read_excel('./data/Mate8.xlsx')
tt=Reviews('content','score',data=data)
tt.drop_invalid(initial_words=initial_words)
tt.segment(product_dict='mobile_dict.txt',stopwords='.\\stopwords\\chinese.txt')
features=tt.get_product_features()
print(features)





