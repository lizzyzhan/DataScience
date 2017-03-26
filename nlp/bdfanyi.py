#coding=utf8

import hashlib
import random
import urllib.parse
import requests
import json

__author__ = 'JSong'


# 具体实现步骤
# 待翻译的文件 每个翻译短语之间 换行
# f=open(path,method)  path 路径 method r read
# while f.readline():
#  f.readline()
# handle translate
# 其实就是 玩 一个翻译的api
# 下面是 百度翻译的api 随便申请下就可以得到
# APP ID: 自己申请
# 密钥: 自己申请
#   http://api.fanyi.baidu.com/api/trans/product/apidoc
# request url   http://api.fanyi.baidu.com/api/trans/vip/translate
'''
支持的语言
auto 	自动检测
zh 	中文
en 	英语
yue 	粤语
wyw 	文言文
jp 	日语
kor 	韩语
fra 	法语
spa 	西班牙语
th 	泰语
ara 	阿拉伯语
ru 	俄语
pt 	葡萄牙语
de 	德语
it 	意大利语
el 	希腊语
nl 	荷兰语
pl 	波兰语
bul 	保加利亚语
est 	爱沙尼亚语
dan 	丹麦语
fin 	芬兰语
cs 	捷克语
rom 	罗马尼亚语
slo 	斯洛文尼亚语
swe 	瑞典语
hu 	匈牙利语
cht 	繁体中文
vie 	越南语
'''


# the const argument
appid = '20170326000043427'
secretKey = 'pSpfYJdraW1prgeB8JvI'
requestUrl='http://api.fanyi.baidu.com/api/trans/vip/translate'
def translate(queryStrings,toLang='en',fromLang='auto',):
    q=queryStrings
    salt=random.randint(32768, 65536)
    sign = appid+q+str(salt)+secretKey
    encodeSign=sign.encode('utf-8')
    m2 = hashlib.md5()
    m2.update(encodeSign)
    sign = m2.hexdigest()
    arguments='?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
    totalUrl=requestUrl+arguments
    r = requests.get(totalUrl)
    if r.text:
        result=json.loads(r.text)
        result=result['trans_result'][0]['dst']
    else:
        result=None               
    return result
#w=translate('taschenlampe','zh','de')
