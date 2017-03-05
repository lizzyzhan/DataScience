from urllib import request
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import re


'''搜索函数
url='https://www.amazon.de/s/?field-keywords=zte'

'''


def amazon_reviews(pid,country='de'):
    #pid='B01D0I0N3C'
    url0="https://www.amazon."+country+"/product-reviews/{product_id}/?pageNumber={pagenum}"
    reviews=[]
    author=[]
    date=[]
    title=[]
    rating=[]
    pagenum=1
    while 1:
        print('begin get review data of page %d'%pagenum)
        url=url0.format(product_id=pid,pagenum=pagenum)
        pagenum+=1
        req = request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1')
        data=request.urlopen(req,timeout=20).read()
        html=BeautifulSoup(data.decode('utf-8','ignore'),'lxml')
        # 评论
        h=html.findAll('span',{'data-hook':'review-body'})
        if not h:
            break
        for hh in h:
            s=hh.contents
            s=[str(t) for t in s if isinstance(t,bs4.element.NavigableString)]
            s=''.join(s)
            s=s.strip()
            s=re.sub(r'[\r\n]+','',s)
            s=re.sub(r'[\n]+','',s)
            reviews.append(s)
        # 作者
        h=html.findAll('a',{'data-hook':'review-author'})
        s=[str(hh.contents[0]) for hh in h]
        author=author+s   
        # 日期
        h=html.findAll('span',{'data-hook':'review-date'})
        s=[str(hh.contents[0]) for hh in h]
        date=date+s
        # 标题
        h=html.findAll('a',{'data-hook':'review-title'})
        s=[str(hh.contents[0]) for hh in h]
        title=title+s  
        # 评分
        h=html.findAll('i',{'data-hook':'review-star-rating'})
        s=[int(hh.contents[0].contents[0].split(',')[0]) for hh in h]
        rating=rating+s
    
    if len(date)==len(author)==len(title)==len(rating)==len(reviews):
        data=pd.DataFrame({'date':date,'author':author,'title':title,'rating':rating,'review':reviews},\
        columns=['date','author','title','rating','review'])
    else:
        data=pd.DataFrame({'review':reviews})
    return data
'''
blade v7 lite:B01D0I0N3C
blade a452:B0148D86AA
blade v7:B01D0I0RN8
'''
data=amazon_reviews('B0148D86AA')
data.to_csv('A452_de_amazon.csv',index=False)
data=amazon_reviews('B01D0I0RN8')
data.to_csv('V7_de_amazon.csv',index=False)