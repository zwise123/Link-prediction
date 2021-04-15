# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:12:23 2019

@author: Administrator
"""

import pandas as pd
import xgboost as xgb
import sys,random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


    
train = pd.read_csv('usa_rw_test_final1.csv')
test = pd.read_csv('usa_rw_train_final1.csv')
 
y,val = train_test_split(train, test_size = 0.1,random_state=1)
test_IMSI = test['src']

y = train.label
val_y = val.label
feature_name_train = [i for i in train.columns[3:]]
feature_name_test = [i for i in test.columns[3:]]
feature_name_val = [i for i in test.columns[3:]]

tests = test[feature_name_test]
trains = train[feature_name_train]
va = val[feature_name_val]


list_rmse = []
def pipeline(iteration,random_seed,max_depth,min_child_weight):
    dtest = xgb.DMatrix(tests)
    dtrain = xgb.DMatrix(trains, label = y)
    dval = xgb.DMatrix(va, label = val_y)
    params={
     	'booster':'gbtree',
        'objective': 'reg:squarederror',
     	'gamma':0.0,
     	'max_depth':max_depth,
     	'lambda':1,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight,  
        'eta': 0.01,
     	'seed':random_seed, 
        'eval_metric':'rmse'
        }
    
    watchlist  = [(dtrain,'train'),(dval,'val')]
    num_round = 600
    model = xgb.train(params, dtrain, num_round, watchlist,verbose_eval=200)
    
    test_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)

    test_result = pd.DataFrame()
    test_result['pre'] = test_y
    test_result['real'] = test['label']
    test_result.to_csv("hepth_call{0}.csv".format(i),index = None,encoding='utf-8')
    y_label = test['label']
    y_score = test_result['pre']
    list_rmse.append(np.sqrt(mean_squared_error(y_label, y_score)))

    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('link_feature_pt_{0}.csv'.format(i),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

  

if __name__ == "__main__":
        
        random_seed = list(range(10000,20000,100))
        max_depth = 4
        subsample = 0.8
        colsample_bytree = 0.9
        min_child_weight = 8
        random.shuffle(random_seed)
        
        for i in range(5):
            print ("iter:", i)
            pipeline(i,random_seed[i],max_depth,min_child_weight)
        print (sum(list_rmse)/5.0)


  