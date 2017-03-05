# coding:utf-8
from multiprocessing.dummy import Pool
import requests
import json
import re
import sys
import os
import csv
from lxml.html import fromstring
from lxml import _elementpath
import pandas as pd

if not os.path.exists("files"):
    os.makedirs('files')

path=sys.path[0]
if os.path.isdir(path):
    mypath=path
if os.path.isfile(path):
    mypath=os.path.dirname(path)
else:
    mypath=os.getcwd()

filepath=mypath+'\\files\\'# 文件保存路径


# 没这header就抓不到
headers = {'Host': 'club.jd.com',
           'Referer': 'http://item.jd.com/0.html'}


def get_jd_rate(pid, pagenum):
    '''!页码从0开始，在网页上显示的第一页'''
    for i in range(20):
        # 因为经常抓到空数据，所以重试20次（本来是while 1）
        try:
            r = requests.get(
                'http://club.jd.com/productpage/p-{}-s-0-t-3-p-{}.html'.format(pid, pagenum), timeout=1, headers=headers)
            if 'content-length' in r.headers:
                # 一般它的值要么是0说明没抓到数据(包括页码超出)，要么不存在
                # print('retry')
                continue
            else:
                #print('.', end=' ')
                return r.text
                # continue
                break
        except Exception as e:
            # print e
            continue
    return ''
    # print(pid, pagenum, 'failed')


def get_jd_rate_totalpagenum(pid):
    # !得到的是pagenum的最大数字，页面上显示的页码，还要+1
    try:
        totalpn = json.loads(get_jd_rate(pid, 0))[
            'productCommentSummary']['commentCount']
        return totalpn // 10
    except:
        # print('failed')
        return -1

def get_jd_all_pid(pid):
    t=requests.get('https://item.jd.com/{}.html'.format(pid),timeout=1).text
    w=re.findall('colorSize:\s{0,3}(\[.*?\])',t)
    product_name=re.findall(u'商品名称：(.*?)</li>',t)
    if product_name:
        product_name=product_name[0]
    else:
        product_name=''
    if not w:
        pid_list=[pid]
        pid_ind=None
    if w:
        w=w[0]
        v=eval(w)
        pid_ind=pd.DataFrame(v)
        pid_list=list(pid_ind['SkuId'])
        pid_list=[t['SkuId'] for t in v]
    return pid_list,product_name,pid_ind


def get_jd_rate_all(pid):
    maxpn = get_jd_rate_totalpagenum(pid)
    if maxpn == -1:
        # print('null')
        return ''
    pp = Pool(100)
    result = pp.map(
        lambda x: get_jd_rate(x[0], x[1]), list(zip([pid] * (maxpn + 1), range(maxpn + 1))))
    pp.close()
    pp.join()
    #pcolor=re.findall(r'productColor":"(.*?)"', str(result))
    data=[]
    for u in result:
        if len(u)<10:
            continue
        try:
            u0=json.loads(u)
            u0=u0['comments']
        except Exception as e:
            print('json.loads fail,the reason is :')
            print(e)
            continue
        for uc in u0:
            guid=uc['guid']
            if 'creationTime' in uc:
                creationTime=uc['creationTime']
            else:
                creationTime=''
            if 'days' in uc:
                days=uc['days']
            else:
                days=''
            if 'content'in uc:
                content=uc['content']
                content=content.strip()
                content=re.sub(r'[\r\n]+','',content)
                content=re.sub(r'[\n]+','',content)
            else:
                content=''

            if 'nickname' in uc:
                nickname=uc['nickname']
            else:
                nickname=''

            if 'productColor' in uc:
                productColor=uc['productColor']
            else:
                productColor=''

            if 'referenceId' in uc:
                referenceId=uc['referenceId']
            else:
                referenceId=''

            if 'referenceName' in uc:
                referenceName=uc['referenceName']
            else:
                referenceName=''

            if 'referenceTime' in uc:
                referenceTime=uc['referenceTime']
            else:
                referenceTime=''

            if 'score' in uc:
                score=uc['score']
            else:
                score=''

            if 'userClient' in uc:
                userClient=uc['userClient']
            else:
                userClient=''

            if 'userLevelId' in uc:
                userLevelId=uc['userLevelId']
            else:
                userLevelId=''

            if 'userLevelName' in uc:
                userLevelName=uc['userLevelName']
            else:
                userLevelName=''

            if 'userProvince' in uc:
                userProvince=uc['userProvince']
            else:
                userProvince=''

            if 'userRegisterTime' in uc:
                userRegisterTime=uc['userRegisterTime']
            else:
                userRegisterTime=''

            data.append((pid,guid,creationTime,days,content,nickname,productColor,\
            referenceId,referenceName,referenceTime,score,userClient,userLevelId,userLevelName,\
            userProvince,userRegisterTime))
    colname=['pid','guid','creationTime','days','content','nickname','productColor',\
    'referenceId','referenceName','referenceTime','score','userClient','userLevelId','userLevelName',\
    'userProvince','userRegisterTime']
    filename=filepath+str(pid)+'.csv'
    with open(filename,"w",encoding='utf-8',newline='') as datacsv:
        csvwriter = csv.writer(datacsv,dialect=("excel"))
        csvwriter.writerow(colname)
        csvwriter.writerows(data)
    print('get '+str(len(data))+' data of pid: '+str(pid))
    return data



def get_list_ids(url):
    r = requests.get(url, headers={'Host': 'list.jd.com',
                                   'Referer': 'http://channel.jd.com/jewellery.html'})
    try:
        scode = r.content.decode('utf-8')
    except:
        scode = r.content.decode('gbk')
    xpath = fromstring(scode).xpath
    ids = xpath('//a/@wareid|//i/@name|//a/@data-tag')
    nextpage = xpath('//a[@class="pn-next"]/@href|//a[@class="next"]/@href')
    nextpage = nextpage[0] if nextpage else False
    stopmsg = '已有0人评价' in scode or '0</a>个评论' in scode
    return (ids, nextpage, stopmsg)


def get_list(url):
    r = requests.get(url, headers={'Host': 'list.jd.com',
                                   'Referer': 'http://channel.jd.com/jewellery.html'})
    try:
        scode = r.content.decode('utf-8')
    except:
        scode = r.content.decode('gbk')
    xpath = fromstring(scode).xpath
    title = xpath('/html/head/title/text()')[0]
    title = re.sub('\s.*', '', title)
    result = []
    while 1:
        ids, nextpage, stopmsg = get_list_ids(url)
        print('get %s' % url)
        result += ids
        if stopmsg:
            print('已经出现评价数量为0的商品，程序终止...')
            break
        if not nextpage:
            print('已达最大页码数，程序终止...')
            break

        url = nextpage if nextpage.startswith(
            'http') else 'http://list.jd.com' + nextpage
    with open('./files/%s.txt' % title, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
        print('结果已存入%s.txt' % title)
    print(result)
    return result

def get_jd_info(pid,filename=None):
    import pandas as pd
    print('begin spyder the comments info of :{pid}'.format(pid=pid))
    pid_list,product_name,pid_ind=get_jd_all_pid(pid)
    print('have get the pid_list: '+','.join(['%s'%p for p in pid_list]))
    if not filename:
        filename='{}_comments.csv'.format(product_name)
    sample_len=0#样本总数
    d=pd.DataFrame()#存储清洗过后的数据
    for pid in pid_list:
        try:
            get_jd_rate_all(pid)
        except Exception as e:
            print('pid ='+str(pid)+' 爬取出现错误')
            print(e)
            continue
        d1=pd.read_csv('{filepath}{pid}.csv'.format(filepath=filepath,pid=pid))
        sample_len+=len(d1)
        d1=d1.drop_duplicates()
        d=d.append(d1)
    # 筛选
    del d1
    # 解决数据中pid和referenceId数据类型为float的情况
    format=lambda x:'%.0f'%x
    d[['pid','referenceId']]=d[['pid','referenceId']].applymap(format)
    index_true=[int(p) in pid_list for p in d['referenceId']]
    d=d[index_true]
    # 筛选出相同评论
    d=d.drop_duplicates(['guid'])
    sample_len1=len(d)
    rate=sample_len1*1.0/sample_len

    # 增加产品信息
    if type(pid_ind)==pd.core.frame.DataFrame:
        colnames=list(pid_ind.columns)
        colnames.remove('SkuId')
        for colname in colnames:
            d[colname]=d['referenceId']
            SkuId=['%s'%p for p in pid_ind['SkuId']]
            referenceId_ind=dict(zip(SkuId,pid_ind[colname]))
            d[colname].replace(referenceId_ind,inplace=True)
    # 修改错误的referenceName
    #referenceName=d['referenceId'].replace(pid_index)
    #d['referenceName']=referenceName
    d.to_csv(filename,index=False)
    return (sample_len1,rate)


def merge_info(pid_list,filapath='.\\files\\',filename=None):
    import pandas as pd
    if not filename:
        filename='{}_comments.csv'.format(pid_list[0])
    sample_len=0#样本总数
    d=pd.DataFrame()#存储清洗过后的数据
    for pid in pid_list:
        d1=pd.read_csv('{filepath}{pid}.csv'.format(filepath=filepath,pid=pid))
        sample_len+=len(d1)
        d1=d1.drop_duplicates()
        d=d.append(d1)
    # 筛选
    del d1
    # 解决数据中pid和referenceId数据类型为float的情况
    format=lambda x:'%.0f'%x
    d[['pid','referenceId']]=d[['pid','referenceId']].applymap(format)
    index_true=[int(p) in pid_list for p in d['referenceId']]
    d=d[index_true]
    # 筛选出相同评论
    d=d.drop_duplicates(['guid'])
    sample_len1=len(d)
    rate=sample_len1*1.0/sample_len
    d.to_csv(filename,index=False)
    return (sample_len1,rate)




def main(url):
    pid = re.findall('jd\.com/(\d+)\.htm', url)
    if pid:
        print(pid[0])
        with open('./files/%s.txt' % pid[0], 'w', encoding='utf-8') as f:
            f.write(get_jd_rate_all(pid[0]))
            print('%s已完成，结果已存入%s.txt' % (pid[0], pid[0]))

    else:
        print('start the mission for product-list pages...')
        get_list(url)

def temp(mylist):
    myset = set(mylist)
    print("total {n} ".format(n=len(myset)))
    for item in myset:
        print("the {item} has found {n}" .format(item=item,n=mylist.count(item)))


# print(main('http://list.jd.com/list.html?cat=6144,6167,6173'))
#while 1:
#    print('\n' + '=' * 80 + '\n')
#    try:
#        command = input(
#            '为了避免拿来主义，本程序功能仅限以下内容：\n1. 输入单个商品页地址或商品ID========>导出评论(pid.ini)\n2. 输入商品列表页地址（比如某类目，不是搜索结果页，事先最好按评论数排序）========>该类目下有评论的ID（类目标题.txt）\n3. 输入文件名（类目标题.txt）========>得到该类目下所有商品评论（类目.ini）\n4. 输入exit或quit退出程序\n注：所有非程序文件读写都在files目录下\n以上模式自动识别，请输入指令：\n')
#        if command == 'exit' or command == 'quit':
#            print('程序结束...')
#            break
#        if command.isalnum():
#            print(command)
#            with open('./files/%s.txt' % command, 'w', encoding='utf-8') as f:
#                f.write(get_jd_rate_all(command))
#                print('%s已完成，结果已存入%s.txt' % (command, command))
#                continue
#        if '.txt' in command:
#            with open('./files/' + command) as ff:
#                ids = ff.read().split()
#            fname = command.replace('.txt', '.ini')
#            print(ids)
#            zongshu = len(ids)
#            jishu = 0
#            with open('./files/' + fname, 'w', encoding='utf-8') as f:
#                for i in ids:
#                    f.write(get_jd_rate_all(i) + '\n')
#                    jishu += 1
#                    print('%s已完成-%s/%s' % (i, jishu, zongshu))
#                    print('所有结果已存入%s' % fname)
#            continue
#
#        main(command)
#    except Exception as e:
#        print(e)
#        print('错误..')

#pid0=[2951490,2473905,2473558,10429992152,10429992153,2600210,2600254,2600258,\
#3196118,3196106,2928957,3169108,3707736,3707756,2174898,2174884,3020827,3237234,\
#3245084,2967927,3478880,3404251,3278880,3245078,2967929,3564140,3717578,\
#3742080,3742076,3742098,3742084,3311989,3742100]
#for i in [1,2,3,4,5,6]:
#    for j in [1,3,5,7,9]:
#        pid0.append(31338*100+i*10+j)
#pid0=pid0+list(range(1856581,1856593))+[3726844,3296831,3726828,3296817]

#pid0=[1856583]+list(range(1856585,1856593))+[3726844,3296831,3726828,3296817]


#pid0=[]
#for i in [1,2,3,4,5,6]:
#    for j in [1,3,5,7,9]:
#        pid0.append(31338*100+i*10+j)



'''
pid=2969549
r=get_jd_info(pid)
print(r)
'''

while 1:
    print('\n' + '=' * 80 + '\n')

    try:
        command = input(
            '本程序具备以下内容：\n\
            1. 输入单个手机ID========>导出评论(_comments.csv)\n\
            2. 输入exit或quit退出程序\n注：所有中间数据读写都在files目录下\n以上模式自动识别，请输入指令：\n')
        if command == 'exit' or command == 'quit':
            print('程序结束...')
            break
        if command.isalnum():
            n,r=get_jd_info(command)
            print(u'一共获取有效评论 %s 条，刷单率为%.2f, 结果已存入 _comments.csv'%(n,r))
            continue
        if '.txt' in command:
            with open('./files/' + command) as ff:
                ids = ff.read().split()
            fname = command.replace('.txt', '.ini')
            print(ids)
            zongshu = len(ids)
            jishu = 0
            with open('./files/' + fname, 'w', encoding='utf-8') as f:
                for i in ids:
                    f.write(get_jd_rate_all(i) + '\n')
                    jishu += 1
                    print('%s已完成-%s/%s' % (i, jishu, zongshu))
                    print('所有结果已存入%s' % fname)
            continue

        main(command)
    except Exception as e:
        print(e)
        print('错误..')





