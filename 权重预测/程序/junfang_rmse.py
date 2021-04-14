# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:15:01 2019

@author: Administrator
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

num_list = []
for i in range(10):
    num_list.append(i)

rmse_list = []    
for qq in num_list:
#data = pd.read_csv('netw_wcnaara_test_final.csv')
    data = pd.read_csv('geomw_wcnaara_test_final1'+str(qq)+'.csv')
    real = data['label']
    predic1 = data['WRA']
#predic2 = data['WAA']
#predic3 = data['WRA']

    mse1 = mean_squared_error(real,predic1)
    rmse1 = np.sqrt(mse1)
    print ('WAA RMSE:',rmse1)
    rmse_list.append(rmse1)
    
ave = sum(rmse_list)/10.0
print ('average rmse: ',ave)
