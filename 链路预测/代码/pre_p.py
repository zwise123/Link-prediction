# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:16:49 2021

@author: 张玮
"""


#传统链路预测使用模体结构进行预测
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

    
test = pd.read_csv('C:/Users/张玮/曹-代码重构/Link-prediction/链路预测/代码/netsciencew_linkmorj_test_final1.csv')
label = test['label']

pre1 = test['M1'] #特征名
pre2 = test['M2']
auc1 = roc_auc_score(label,pre1)
auc2 = roc_auc_score(label,pre2)
print('auc_M1:',auc1)
print('auc_M2:',auc2)
