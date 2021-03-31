# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:58:48 2021

@author: 张玮
"""


#特征选择
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

all_postitive_AUC = []
for i in [5,6,7,8,9]:
    auc10times = []
    for num in range(20):
        
        