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


def jieba_cut(texts,userdict=None,stopwords=None,POS=False,add_words=[]):
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
    '''

    '''
    
    '''结巴分词词性对照表
    a:形容词,ad 副形词,an 名形词,ag 形容词性语素,al 形容词性惯用语
    b:区别词,bl 区别词性惯用语
    c:连词,cc 并列连词
    d:副词
    e:叹词
    f:方位词
    h:前缀
    k:后缀
    m:数词,mq 数量词
    n:名词,nr 人名,ns 地名,nt 机构团体名,nz 其它专名,nl 名词性惯用语,ng 名词性语素
    o 拟声词
    p 介词,pba 介词“把”,pbei 介词“被”
    q 量词,qv 动量词,qt 时量词
    r 代词,rr 人称代词,rz 指示代词,ry 疑问代词,rg 代词性语素
    s 处所词,t 时间词,u 助词,v 动词,w 标点符号,x 字符串,y 语气词(delete yg),z 状态词
    '''
    userdict='mobile_dict.txt'
    stopwords='.\\stopwords\\chinese.txt'
    add_words=color
    POS=True
    texts=tt.texts_raw


    if  userdict is not None:
        jieba.load_userdict(userdict)
    if isinstance(stopwords,str):
        f=open(stopwords,encoding='utf-8')
        stopwords=f.readlines()
        stopwords=[s.strip() for s in stopwords]

    if isinstance(texts,pd.core.series.Series):
        index=texts.index
    else:
        index=range(len(texts))

    texts='燚燚燚'.join(texts)# 为了分词用
    # 手机内存、屏幕尺寸等
    add_words+=list(set(re.findall('[\d\.\+]{1,5}[g寸万]{1}',texts)))
    # 识别品牌名称+型号，如红米4a，骁龙835
    brand_name=['小米','红米','华为','荣耀','mate','三星','苹果','oppo','vivo',\
    '魅族','锤子','坚果','美图手机','美图','联想','高通骁龙','骁龙','联发科']
    for b in brand_name:
        add_words=add_words+list(set(re.findall(b+'[\da-z]*',texts)))
    for word in add_words:
        jieba.add_word(word,tag='n')


    # 新方法，为了能有识别新词能力，我们把语料合并分词(后期可以按照一万个评论一批)
    jieba.add_word('燚燚燚')
    # 去掉分词结果中一份评论全是字母或数字的情况
    pattern=re.compile(r'\s[a-z\.]+\s|^[a-z\.]+\s|\s[a-z\.]+$|\s[\d\.]+\s|^[\d\.]+\s|\s[\d\.]+$')
    if POS:
        tmp=jieba.posseg.lcut(texts)
        texts=' '.join([w.word for w in tmp if w.word not in stopwords]).split('燚燚燚')
        texts=list(map(lambda x:re.sub(r'\s+',' ',re.sub(pattern,' ',x).strip()),texts))
        texts=pd.Series(texts,index=index)
        words_pos=dict(tmp)
        return texts,words_pos
    else: 
        texts=' '.join([w for w in jieba.cut(texts) if w not in stopwords]).split('燚燚燚')
        texts=list(map(lambda x:re.sub(r'\s+',' ',re.sub(pattern,' ',x).strip()),texts))
        texts=pd.Series(texts,index=index)
        return texts





class Reviews():
    """初始化"""
    def __init__(self, contents=None,scores=None,data=None,language=None):
        '''初始化需要两个数据，一份是评论，一份是评分
        如果给定data,则contents和scores是对应的列名字，否则contents是具体的数组
        '''
        if data is not None:
            '''此时contents和scores肯定存在'''
            texts=data[contents]
            scores=data[scores]     
        if scores is not None:
            scores=pd.Series(scores)
        if contents is not None:
            texts=pd.Series(contents)
        if texts is not None:
            texts=texts.map(lambda x:self.cleaning(x))
            ind=texts.map(lambda x:len('%s'%x)>2)
            texts=texts[ind].reset_index(drop=True)
        if scores is not None:
            scores=scores[ind].reset_index(drop=True)       
        self.texts_raw=texts#原始语料
        self.language=language#暂时只适配中文，后期会适配英文、德文等
        if language in ['english','en']:
            # 部分语系不需要分词
            self.texts_seg=self.texts_raw
        else:
            self.texts_seg=None
        self.texts_vec_idf=None#(向量化稀疏数组,words)
        self.texts_vec_tf=None#(向量化稀疏数组,words)
        self.scores=scores
        self.pos={}


    def cleaning(self,text):
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



    def find_keywords(self,keywords,texts=None):
        '''根据给定的关键词在评论预料中查找到相应的句子（非整条评论）
        该函数被设计为全局函数，即可直接使用：Reviews.find_keywords('拍照',texts)
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
        if (self.texts_raw is None) and (texts is not None):
            texts_keywords=texts.map(_keywords_find)
        else:
            texts_keywords=self.texts_raw.map(_keywords_find)
        texts_keywords=texts_keywords[texts_keywords.notnull()]
        return texts_keywords


    def _isrubbish(self,x,keywords):
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

    
    def drop_invalid(self,initial_words=[],max_rate=0.8,show_invalid=False):
        '''去除无效评论
        1、根据给定的初始无效关键词，寻找所有相似的词语
        2、统计无效关键词的占比，大于max_rate的则判定为无效评论
        parameter
        --------
        initial_words: 初始的无效评论关键词
        max_rate: 一个评论中，如果垃圾关键词占比超过max_rate,则会判定为无效
        show_invalid: 如果为真，则函数返回 self,texts_invalid
        
        return
        ------
        self
        返回的是逻辑索引
        '''
        if self.texts_seg is None:
            print('该对象还未分词，请检查')
            return self
        texts_vec,words=self.vectorizer(store=False)
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
                a,b=np.where(tmp>=max_rate)
                similar_words+=[words[i] for i in a]
        invalid=self.texts_seg.map(lambda x:self._isrubbish(x,similar_words))
        if show_invalid:
            texts_invalid=self.texts_raw[invalid]
        self.texts_raw=self.texts_raw[~invalid]
        if id(self.texts_raw) != id(self.texts_seg):
            self.texts_seg=self.texts_seg[~invalid]
        if self.scores is not None:
            self.scores=self.scores[~invalid]
        if show_invalid:
            return self,texts_invalid
        else:
            return self

    def replace(self,synonyms):
        '''同义词等替换，如将老爸替换成爸爸，apple替换成苹果之类的
        words_dict: dict字典或者text文件路径
        '''
        if isinstance(synonyms,str) and os.path.exists(synonyms):
            f=open(synonyms,encoding='utf-8')
            add_words=f.readlines()
            synonyms=dict(list(map(lambda x:re.sub('：',':',x.strip()).split(':'),add_words[1:])))

        def _sub_replace(text):
            for k in synonyms:
                text=re.sub(k,synonyms[k],text)
            return text
        self.texts_raw=self.texts_raw.map(_sub_replace)
        return self 
    

    def segment(self,product_dict=[],stopwords=[],add_words=[]):
        '''
        对中文文本进行分词，并将分词结果用空格隔开
        parameter
        ---------
        product_dict: 自己添加的 jiaba 词典
        stopwords：停止词，用于分词
        
        具体参见函数：jieba_cut
        '''
        texts_seg,words_pos=jieba_cut(self.texts_raw,userdict=product_dict,stopwords=stopwords,POS=True,add_words=add_words)
        self.texts_seg=texts_seg
        self.pos=words_pos
        return self


    def vectorizer(self,vec_model='idf',select_features=None,store=True,**kwargs):
        '''对语料向量化
        vec_model: tfidf/idf、tf、word2vec等
        对象默认会存储idf和tf分词结果，便于复用
        '''
        if vec_model in ['idf','tfidf']:
            myvectorizer = TfidfVectorizer(min_df=2,max_df=0.95,ngram_range=(1,2),sublinear_tf=True,**kwargs)
        else:
            myvectorizer = CountVectorizer(**kwargs)
        texts_vec = myvectorizer.fit_transform(self.texts_seg)
        #texts_vec=texts_vec.toarray()
        words=np.array(myvectorizer.get_feature_names())
        if select_features is not None:
            n_features_initial=texts_vec.shape[1]
            n_features=min(n_features_initial,select_features)
            if 'score_func' in kwargs:
                score_func=kwargs['score_func']
            else:
                score_func=f_classif
            selector = SelectKBest(score_func,k=n_features).fit(texts_vec,self.scores)
            informative_words_index = selector.get_support(indices=True)
            words = np.array([words[i] for i in informative_words_index])
            texts_vec=texts_vec[:,informative_words_index]
        if store and (vec_model in ['idf','tfidf']):
            self.texts_vec_idf=(texts_vec,words)
        elif store and (vec_model in ['tf']):
            self.texts_vec_tf=(texts_vec,words)
        return texts_vec,words
 
    def get_keywords(self,condition=None,topK=20):
        ''' 返回语料的关键词，算法是textrank
        '''
        if condition is not None:
            texts=' '.join(self.texts_raw[condition])
        else:
            texts=' '.join(self.texts_raw)
         
        keywords=jieba.analyse.textrank(texts,topK=topK)
        return keywords



    def dis_of_pairwords(self,pair_words,texts=None):
        '''返回两个词在评论预料中的距离
        用于检测两个词是否常常放在一起用，例如手机、便宜会出现在一起，但拍照、便宜就不会出现在一起
        parametre
        --------
        s:['手机','不错']
        texts: 语料
        '''
        if self.texts_raw is not None:
            texts=self.texts_raw
        dis=[]
        pattern=re.compile(pair_words[0]+'.*?'+pair_words[1]+'|'+pair_words[1]+'.*?'+pair_words[0])
        for text in texts:
            tmp=re.findall(pattern,text)
            dis0=len(tmp[0])-len(pair_words[0])-len(pair_words[1]) if tmp else np.nan
            dis.append(dis0)
        dis=pd.Series(dis)
        dis=dis[dis.notnull()].quantile(0.05,'nearest')
        return dis



    def get_product_features(self,min_support=0.005,max_dis_pairwords=4):
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
            self.vectorizer(vec_model='tf')
        texts_vec_tf,words=self.texts_vec_tf
        # 第一步，利用关联分析找到初始的特征词
        gen=associate.frequent_itemsets(texts_vec_tf, min_support=min_support)
        sup=[[words[i] for i in s[0]] for s in gen if len(s[0])==2]
        # 第二步 筛选出 一个为名词，一个为形容词的特征，同时将名词排前面
        sup=[sorted(s,key=lambda x:self.pos[x][0]!='n') for s in sup if set([self.pos[s[0]],self.pos[s[1]][0]])==set(['n','a'])]   
        # 第三步 从中选出名词和形容词配对那些特征词
        sup=[kw for kw in sup if self.dis_of_pairwords(kw)<max_dis_pairwords]     
        features_frequent=list(set([s[0] for s in sup]))
        opinion_words=list(set([s[1] for s in sup]))

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


    def features_sentiments(self,features=None,thr=0.5):
        '''产品特征词的情感分析
        parameter
        --------
        features: 产品特征词，如：内存、续航、拍照等
        thr：情感判断，当大于thr时，判定为正面意见。默认为0.5
        return
        ------
        features_opinion：DataFrame 格式，各个特征词的提及数、正面意见占比，负面意见占比
        features_corpus：dict格式，原始语料中提及到特征的句子以及对应的情感分析
        '''
        if features is None:
            features=self.get_product_features()
        features_opinion=pd.DataFrame(index=features,columns=['total','mention_count','p_positive','p_negative'])
        features_corpus={}
        N=len(self.texts_raw)
        for fw in features:
            texts_fw=self.find_keywords(fw)
            features_corpus[fw]=texts_fw
            #features_corpus[fw]=' || '.join(texts_fw)
            sc=texts_fw.map(lambda x:SnowNLP(x).sentiments)
            features_corpus[fw]=pd.concat([texts_fw,sc],axis=1)
            features_corpus[fw].columns=['sentences','sentiments']
            p=len(sc[sc>thr])/len(sc) if len(sc)>0 else np.nan
            features_opinion.loc[fw,:]=[N,len(sc),p,1-p]   
            #features_opinion[fw]=(len(sc),'{:.2f}%'.format(p*100),'{:.2f}%'.format(100-p*100))
        return features_opinion,features_corpus

    def find_topic(self,condition=None,n_topics=10,n_words=10,topic_model='lda',vec_model='tf',show=True,**kwargs):
        '''主题模型，和上面那个函数，优先使用该函数
        parameter
        ---------
        condition: 语料逻辑值，可以用于专门对好评/差评进行主题分解
        n_topics: 主题数
        n_words: 每个主题输出的词语数
        vec_model: 向量化方法，默认是tf
        '''
        if condition is not None:
            texts=self.texts_seg[condition]
        else:
            texts=self.texts_seg
        if topic_model in ['lda','LDA']:
            dictionary = corpora.Dictionary([doc.split(' ') for doc in texts])
            corpus = [dictionary.doc2bow(text.split(' ')) for text in texts]
            if vec_model in ['idf','tfidf']:
                tfidf = models.TfidfModel(corpus)
                corpus = tfidf[corpus]
            lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)
            topics_keywords=lda.show_topics(num_topics=n_topics, num_words=n_words,formatted=False)
            if show:
                print('\n'.join(['主题 {}: {}'.format(i,' | '.join([k[0] for k in \
                topic[1]])) for i,topic in enumerate(topics_keywords)]))
            return topics_keywords

    def genwordcloud(self,condition=None,filename='评论词云.png',mask=None,background_color='white',\
    font_path='DroidSansFallback.ttf',**kwargs):
        '''生成词云
        # mask 是RGBA模式，最后一个分量是alpha通道, 如：mask_circle.png
        '''
        from PIL import Image
        from wordcloud import WordCloud
        
        if condition is not None:
            contents=self.texts_seg[condition]
        else:
            contents=self.texts_seg
        contents='. '.join(contents)
        if mask is None:
            tmp=np.zeros((900,1200),dtype=np.uint8)
            for i in range(tmp.shape[0]):
                for j in range(tmp.shape[1]):
                    if (i-449.5)**2/(430**2)+(j-599.5)**2/(580**2)>1:
                        tmp[i,j]=255
            mask=np.zeros((900,1200,4),dtype=np.uint8)
            mask[:,:,0]=tmp
            mask[:,:,1]=tmp
            mask[:,:,2]=tmp
            mask[:,:,3]=255
        else:
            mask=np.array(Image.open(mask))
        wordcloud = WordCloud(background_color = background_color,font_path=font_path, mask = mask)
        wordcloud.generate(contents)
        wordcloud.to_image().save(filename)
        keywords=[(w[0][0],w[0][1],w[1],w[2][0],w[2][1]) for w in wordcloud.layout_]
        comments_keys=pd.DataFrame(keywords,columns=['key','count','font_size', 'position_x','position_y'])
        return comments_keys


# ======================================================================
initial_words=['女娲造人','混沌初开','天崩地裂','永不变心','寝食难安','七彩祥云',\
'七经八脉', '焚香祷告','凑齐银两','呜呼哀哉','海枯石烂','阅商无数','顶天立地','欣喜若狂',\
'乃是','天上','三斧','小生', '七彩', '祥云','吾','宝物','金光四射','金甲','天神']

data=pd.read_excel('./data/红米4A.xlsx')
color=list(data['productColor'].dropna().unique())
tt=Reviews(data['content'],data['score'])
tt.replace('synonyms.txt')
tt.segment(product_dict='mobile_dict.txt',stopwords='.\\stopwords\\chinese.txt',add_words=color)
tt.drop_invalid(initial_words=initial_words,max_rate=0.6)
features=tt.get_product_features()
print(features)
features1=list(set(features)-set(['京东','物流']) | set(['拍照']))
features_opinion,features_corpus=tt.features_sentiments(features=features,thr=0.5)

# 好中差评论分析
cc={'好评':tt.scores.isin([5]),'中评':tt.scores.isin([3,4]),'差评':tt.scores.isin([1,2])}
for c in cc:
    #topics_keywords=tt.find_topic(tt.scores.isin([5]))
    keywords1=tt.get_keywords(cc[c])
    print(c+'关键词：')
    print('|'.join(keywords1))
    tt.genwordcloud(cc[c],filename=c+'词云.png')
    


'''
# ==================[自动摘要算法]======================
from textrank4zh import TextRank4Keyword, TextRank4Sentence
tr4w = TextRank4Keyword()
tr4w.analyze(text=' '.join(tt.texts_raw), lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
print( '关键词：' )
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)
print()
print( '关键短语：' )
for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num= 2):
    print(phrase)
tr4s = TextRank4Sentence()
tr4s.analyze(text=' '.join(tt.texts_raw), lower=True, source = 'all_filters')
print()
print( '摘要：' )
for item in tr4s.get_key_sentences(num=3):
    print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重
'''





