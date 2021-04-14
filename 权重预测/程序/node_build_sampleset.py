# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:34:30 2017

@author: dell
"""

import pandas as pd
import networkx as nx
import random
from sklearn import preprocessing
import numpy as np
import math

 
G = nx.read_edgelist('权重预测/数据/权重归一化后数据/netscience_weight2.txt', nodetype=int, data=(('weight',float),))
    
G = G.to_undirected()

edge_all = list(G.edges())
ebunch_edges = edge_all
random.shuffle(ebunch_edges)

pos_src = []
pos_dst = []
pos_weight = []

for i in ebunch_edges:
    pos_src.append(i[0])
    pos_dst.append(i[1])
    pos_weight.append(G[i[0]][i[1]]['weight'])
       
#################训练集#####################
            
pos_num = int(len(pos_src)*0.1)    #取10%的样本数
           
src_train = pos_src[pos_num:int(len(pos_src))]
dst_train = pos_dst[pos_num:int(len(pos_src))]
label_train = pos_weight[pos_num:int(len(pos_src))]

usair = pd.DataFrame()
usair['src'] = src_train
usair['dst'] = dst_train
usair['label'] = label_train
usair.to_csv('权重预测/net_train_random.csv',index = False)
####################测试集#################################
           
src = pos_src[:pos_num] 
dst = pos_dst[:pos_num]
label = pos_weight[:pos_num] 

usair = pd.DataFrame() 
usair['src'] = src
usair['dst'] = dst
usair['label'] = label
usair.to_csv('权重预测/net_test_random.csv',index = False)      
