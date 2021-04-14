# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:58:48 2021

@author: 张玮
"""


from numpy import loadtxt
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from matplotlib import pyplot
from scipy.stats import pearsonr
import xgboost as xgb
import random
import copy
import networkx as nx
import pandas as pd
import numpy as np
import os
import itertools
import math
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import f_regression
######################

all_positive_AUC = []

for i in [5,6,7,8,9]:
    auc10times = []
    for num in range(20):
        data1=pd.read_csv('netsciencew_linkmorj_test_final'+str(i)+str(num)+'.csv')
        data2=pd.read_csv('netsciencew_linkmorj_train_final'+str(i)+str(num)+'.csv')
        
        positive_predictor = ['three_one','three_two','four_one','four_two','four_three','four_four','four_five1','four_six','four_two2','four_three2','four_three3','four_five2']
        
        xgbc=XGBClassifier()
        
        xgbc.fit(data2[positive_predictor],data2['label'])
        test_result=xgbc.predict_proba(data1[positive_predictor])[:,1]
        
        auc = roc_auc_score(data1['label'], test_result)
        
        auc10times.append(auc)
    all_positive_AUC.append(np.mean(auc10times))
print(all_positive_AUC)
      
   
positive_SelectFromModel_AUC = []
for i in [5,6,7,8,9]:
    auc10times = []
    for num in range(20):
        data1=pd.read_csv('netsciencew_linkmorj_test_final'+str(i)+str(num)+'.csv')
        data2=pd.read_csv('netsciencew_linkmorj_train_final'+str(i)+str(num)+'.csv')
        
        positive_predictor = ['three_one','three_two','four_one','four_two','four_three','four_four','four_five1','four_six','four_two2','four_three2','four_three3','four_five2']
        
        X_train = data2[positive_predictor]
        y_train = data2['label']
        X_test = data1[positive_predictor]
  
        
        xgbc=XGBClassifier()
        
        xgbc.fit(X_train,y_train)
        print("clf.feature_importances_ :",xgbc.feature_importances_)
        selection = SelectFromModel(xgbc, threshold=0.14,prefit=True)  
        
        select_X_train = selection.transform(X_train)
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        select_X_test = selection.transform(X_test)
        print ('new_train:',select_X_train.shape)
        print ('new_test:',select_X_test.shape)
        result = selection_model.predict_proba(select_X_test)[:,1]
        
        auc = roc_auc_score(data1['label'], result)
        auc10times.append(auc)
    positive_SelectFromModel_AUC.append(np.mean(auc10times))     

'''
positive_SelectFromModel_AUC = []
#positive_feature_importance_precision = []
data1=pd.read_csv('netsciencew_linkmorj_test_final919.csv')
data2=pd.read_csv('netsciencew_linkmorj_train_final919.csv')
        
positive_predictor = ['three_one','three_two','four_one','four_two','four_three','four_four','four_five1','four_six','four_two2','four_three2','four_three3','four_five2']
        
X_train = data2[positive_predictor]
y_train = data2['label']
X_test = data1[positive_predictor]
  
        
xgbc=XGBClassifier()
        
xgbc.fit(X_train,y_train)
print("clf.feature_importances_ :",xgbc.feature_importances_)
selection = SelectFromModel(xgbc, threshold=0.085,prefit=True)  

select_X_train = selection.transform(X_train)
selection_model = XGBClassifier()
selection_model.fit(select_X_train, y_train)
select_X_test = selection.transform(X_test)
print ('new_train:',select_X_train.shape)

result = selection_model.predict_proba(select_X_test)[:,1]
        
auc = roc_auc_score(data1['label'], result)
#            print (auc,precision)
print ('auc',auc)
 
''' 
    
positive_chi2_AUC = []
for i in [5,6,7,8,9]:
    auc10times = []
    for num in range(20):
        data1=pd.read_csv('netsciencew_linkmorj_test_final'+str(i)+str(num)+'.csv')
        data2=pd.read_csv('netsciencew_linkmorj_train_final'+str(i)+str(num)+'.csv')
        
        positive_predictor = ['three_one','three_two','four_one','four_two','four_three','four_four','four_five1','four_six','four_two2','four_three2','four_three3','four_five2']
        
        X_train = data2[positive_predictor]
        y_train = data2['label']
        X_test = data1[positive_predictor]
  
        
        xgbc=XGBClassifier()
        
        ch2 = SelectKBest(chi2,k=5)
        
        select_X_train = ch2.fit_transform(X_train, y_train)
        select_X_test = ch2.transform(X_test)

        xgbc.fit(select_X_train,y_train)
        
        result = xgbc.predict_proba(select_X_test)[:,1]
        
        auc = roc_auc_score(data1['label'], result)
        auc10times.append(auc)
    positive_chi2_AUC.append(np.mean(auc10times))     
print (positive_chi2_AUC)
  

positive_RFE_AUC = []
for i in [5,6,7,8,9]:
    auc10times = []
    for num in range(20):
        data1=pd.read_csv('netsciencew_linkmorj_test_final'+str(i)+str(num)+'.csv')
        data2=pd.read_csv('netsciencew_linkmorj_train_final'+str(i)+str(num)+'.csv')
        
        positive_predictor = ['three_one','three_two','four_one','four_two','four_three','four_four','four_five1','four_six','four_two2','four_three2','four_three3','four_five2']
        
        X_train = data2[positive_predictor]
        y_train = data2['label']
        X_test = data1[positive_predictor]
  
        xgbc=XGBClassifier()
        estimator = XGBClassifier()
        selector = RFE(estimator=estimator, n_features_to_select=5)
        select_X_train = selector.fit_transform(X_train, y_train)
        select_X_test = selector.transform(X_test)
        
        xgbc.fit(select_X_train, y_train)
        result = xgbc.predict_proba(select_X_test)[:,1]
        
        auc = roc_auc_score(data1['label'], result)
        auc10times.append(auc)
    positive_RFE_AUC.append(np.mean(auc10times))     
    
    
    
   
df = pd.DataFrame()
df['train_size'] = [5,6,7,8,9]
df['all_positive_AUC'] = all_positive_AUC
df['positive_SelectFromModel_AUC'] = positive_SelectFromModel_AUC 
df['positive_chi2_AUC'] = positive_chi2_AUC 
df['positive_RFE_AUC'] = positive_RFE_AUC 

df.to_csv('link_selection_accurac5.csv',index = False)
        