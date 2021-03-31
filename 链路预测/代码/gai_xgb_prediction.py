
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:05:14 2021

@author: 张玮
"""


import pandas as pd
from xgboost import XGBClassifier
from sklearn import metrics
import random
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from xgboost import plot_importance


train = pd.read_csv('链路预测/代码/netsciencew_linkmorj_train_final1.csv')    #训练集数据
test = pd.read_csv('链路预测/代码/netsciencew_linkmorj_test_final1.csv')      #测试集数据
feature_name_train = [i for i in train.columns[3:]]    #特征列名
feature_name_test = [i for i in test.columns[3:]]

tests = test[feature_name_test]               #特征数据
trains = train[feature_name_train]

xgbc = XGBClassifier()        #XGBoos没有方任何参数，即使用原始参数
xgbc.fit(trains,train['label'])     #模型拟合，模型训练
    
    
#    plot_importance(xgbc)
#    plt.show()
#    plt.savefig('net_feature_import'+str(i)+'.pdf', dpi=600,bbox_inches='tight')
    
test_result=xgbc.predict_proba(tests)[:,1]        #使用模型进行预测
print(test_result)
result_test = pd.DataFrame()
result_test['reall'] = test['label']
result_test['pre'] = test_result
result_test.to_csv('链路预测/代码/xgb_pre.csv',index = None)

auc = roc_auc_score(test['label'],test_result)    #auc衡量预测准确性 
print ('auc: ',auc)

