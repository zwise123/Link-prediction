# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:35:19 2019

@author: Administrator
"""

import pandas as pd
import networkx as nx
#import igraph as ig
import numpy as np
import random  
import math
from sklearn.metrics import mean_squared_error

num_list = []
for i in range(10):
    num_list.append(i)

rmse_list = []    
for qq in num_list:
    data = pd.read_csv('geomw_rw_test_final1'+str(qq)+'.csv')

    weight_d = data['label']
    rwra_d = data['rWRA']

    weight_array = np.array(list(weight_d))
    rwra_array = np.array(list(rwra_d))

    l = [i/100 for i in range(1,301,1)]

    result = []

    for i in l:
        r = np.linalg.norm(i*rwra_array-weight_array)
        result.append(r)

    index = result.index(min(result))
    b_l = l[index]
    print ('brtter lama = ',b_l)

    rwra_after = b_l*rwra_d

    rmse = np.sqrt(mean_squared_error(weight_d, rwra_after))
    print ('rmse = ',rmse)
    rmse_list.append(rmse)

ave = sum(rmse_list)/10.0
print ('average rmse: ',ave)