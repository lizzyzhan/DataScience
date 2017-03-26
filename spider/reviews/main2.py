# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 11:29:50 2017

@author: gason
"""

import pandas as pd
import amazon
import os
import random

pid_list=pd.read_csv('phone_pid_de_info.csv',index_col='pid')
pid_list=list(pid_list.index)
pid_list=random.sample(pid_list,len(pid_list))


file=os.listdir('.\\info\\')
pid_finished=[os.path.splitext(f)[0] for f in file]

for pid in pid_list:
    file=os.listdir('.\\info\\')
    pid_finished=[os.path.splitext(f)[0] for f in file]
    if pid not in pid_finished:
        data=amazon.get_info(pid)
        print('have get %s'%pid)
    else:
        print('%s exist!'%pid)