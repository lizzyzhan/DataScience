# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 21:01:32 2017

@author: gason
"""

import pandas as pd
import amazon
import os
import random

pid_list=pd.read_csv('pid_info.csv',index_col='pid')
pid_list=list(pid_list.index)
pid_list=random.sample(pid_list,len(pid_list))


file=os.listdir('.\\data\\')
pid_finished=[os.path.splitext(f)[0] for f in file]

for pid in pid_list:
    file=os.listdir('.\\data\\')
    pid_finished=[os.path.splitext(f)[0] for f in file]
    if pid not in pid_finished:
        data=amazon.get_reviews(pid)
        print('have get %s'%pid)
    else:
        print('%s exist!'%pid)











