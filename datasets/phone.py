# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:48:11 2017
手机号码归属地

@author: gason
"""
from bs4 import BeautifulSoup
import urllib.request as urllib2
import re

def phoneregion(phone):
    #phone=1760795
    phone='%s'%phone
    phone=phone[:7]
    url = "http://www.ip138.com:8080/search.asp?mobile=%s&action=mobile" % phone
    r=urllib2.urlopen(url).read().decode('gb2312')
    r=re.sub('<td','<TD',r)
    r=re.sub('td>','TD>',r)
    r=re.sub('=tdc2>','="tdc2">',r)
    r=re.sub('&nbsp;','',r)
    soup = BeautifulSoup(r, "xml")
    s = soup.find_all('TD', class_="tdc2")
    if s:
        region = s[0].contents[-1]
        cardtype = s[1].contents[-1]
        regioncode = s[2].contents[0]
        postcode = s[3].contents[0]
    else:
        region=''
        cardtype=''
        regioncode=''
        postcode=''
    result = {
        "phone": phone,
        "region": region,
        "cardtype": cardtype,
        "regioncode": regioncode,
        "postcode": postcode,
    }
    return result
