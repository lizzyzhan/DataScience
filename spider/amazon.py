# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:51:58 2017

@author: gason
"""

from urllib import request
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import re
import time


'''搜索函数
url='https://www.amazon.de/s/?field-keywords=zte'

'''

global config

config={
'de':{
'price_name':'preis',
'unit':'EUR',
'unit_symbol':'€',
'price_sep':'.'}}







def get_search_list(url):
    #url='https://www.amazon.de/s/rh=i:electronics,n:562066,n:!569604,n:1384526031,n:3468301,p_36:115014031&pf_rd_i=571954&sort=price-desc-rank&page=11111'
    req = request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1')
    log='finish'
    try:
        data=request.urlopen(req,timeout=200).read()
    except:
        pid=[]
        log='time out'
        print('time out')
        return (pid,log)
    html=BeautifulSoup(data.decode('utf-8','ignore'),'lxml')
    try:
        h=html.findAll('div',{'id':"btfResults"})[0]
        t=h.findAll('a')
        t2=[tt for tt in t if tt.has_attr('title')]
        pid=[re.findall('/dp/(.*?)/',tt['href'])[0] for tt in t2]
    except:
        pid=[]
        log='no pid'
    return (pid,log)
  
    
def get_phone_pid_de():
    url_list=[]
    # 500欧元以上的手机
    # &sort=price-desc-rank
    url_list.append('https://www.amazon.de/s/rh=i:electronics,n:562066,n:!569604,\
    n:1384526031,n:3468301,p_36:115014031&pf_rd_i=571954&page={page}')
    # 200-500欧元之间的手机
    url_list.append('https://www.amazon.de/s/rh=i:electronics,n:562066,n:!569604,\
    n:1384526031,n:3468301,p_36:115013031&pf_rd_i=571954&page={page}')
    # 0-200欧元之间的手机
    url_list.append('https://www.amazon.de/s/rh=i:electronics,n:562066,n:!569604,\
    n:1384526031,n:3468301,p_36:0-20000&pf_rd_i=571954&page={page}')
    phone_pid=[]
    errorurl=[]
    for url0 in url_list:
        page=1
        while True:
            url=url0.format(page=page)
            pid,log=get_search_list(url)
            if not pid:
                if log == 'no pid':
                    break
                else:
                    errorurl.append(url)
                    print('网页获取超时，脚本先暂停100秒..........')
                    time.sleep(100)
            page+=1
            phone_pid=phone_pid+pid
            print('page %d: '%(page)+'get %d pid'%(len(phone_pid)))
    return phone_pid,errorurl


def get_info(pid,country='de'):
    #pid='B01D0I0N3C'
    #country='de'
    info={'pid':pid,'country':country,\
    'pname':None,\
    'brand':None,\
    'price':None}
    url='https://www.amazon.{country}/dp/{pid}/'.format(country=country,pid=pid)
    req = request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1')
    try:
        data=request.urlopen(req,timeout=200).read()
    except:
        print('time out')
        return info
    html=BeautifulSoup(data.decode('utf-8','ignore'),'lxml')
    
    # 产品名称
    try:
        pname=html.title.text
        pname=pname.split(':')[0]
    except:
        pname=None            
    # 手机价格
    try:
        h=html.findAll('div',{'id':'price'})
        h1=h[0].findAll('tr')
        price=None
        for hh in h1:
            tmp=hh.findAll('td')
            tmp1=tmp[0].text.lower()
            tmp2=tmp[1].text
            if config['de']['price_name']==tmp1[:-1]:
                tmp2=re.findall('[\d.,]{2,}',tmp2)
                if tmp2:
                    price=tmp2[0]
        if price:
            if country in ['de']:
                price=re.sub('\.','',price)
                price=float(re.sub(',','.',price))
            else:
                price=float(re.sub(',','',price))
    except:
        price=None
    # 手机品牌
    try:
        brand=None
        tmp=html.findAll('a',{'id':'brand'})
        if tmp:
            tmp=tmp[0].text
            brand=re.sub('\s','',tmp)
        if not brand:
            brand=pname.split(' ')[0]
    except:
        brand=None
    info['pname']=pname
    info['brand']=brand
    info['price']=price
    return info
        



def get_reviews(pid,country='de'):
    #pid='B01D0I0N3C'
    country='de'
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
        data=request.urlopen(req,timeout=50).read()
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
#data=amazon_reviews('B0148D86AA')
#data.to_csv('A452_de_amazon.csv',index=False)
#data=amazon_reviews('B01D0I0RN8')
#data.to_csv('V7_de_amazon.csv',index=False)