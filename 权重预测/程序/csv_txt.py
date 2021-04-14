# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:10:27 2019

@author: Administrator
"""

import networkx as nx
import pandas as pd
import numpy as np
import math



data = pd.read_csv('geom_test1.csv')

src = data['src']
dst = data['dst']
weights = data['label']

pos_num = int(len(src))

with open("geom_test1.txt",'w') as f:   #将节点写入，不带标签
    for i in range(pos_num):
        print(src[i], dst[i], weights[i], file=f)

