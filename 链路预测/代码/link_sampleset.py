# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:51:06 2021

@author: 张玮
"""

import networkx as nx
import pandas as pd 
import random 
import numpy as np
import math


#构建网络
g = nx.read_edgelist('../数据/netscience_weight2.txt', nodetype=int, data=(('weight',float),))
for edge in g.edges():
    g[edge[0]][edge[1]]['weight'] = 1
g = g.to_undirected()


#打乱边
edge_all = list(g.edges())
ebunch_edges = edge_all
random.shuffle(ebunch_edges)

#训练集中的正样本为网络中连边的90%，测试集中的正样本为网络连边中剩余的10%
#从网络中不存在的连边中选取与训练集和测试集正样本相同数量的负样本分别作为训练集和测试集的负样本
#最终要使得训练集和测试集中的负样本的比例为1：1

#构建正样本
pos_src = []
pos_dst = []
pos_weight = []
pos_label = []

for i in ebunch_edges:
    if i[0] != i[1]:
        pos_src.append(i[0])
        pos_dst.append(i[1])
        pos_label.append(1)
        
pos_w = int(len(pos_src))
print(pos_w)

#######构建负样本##############
neg_src = []
neg_dst = []
#neg_weight = []
neg_label = []
us_edges = zip(pos_src,pos_dst) 

first_src = list(set(pos_src[int(len(pos_src)*0.05):int(len(pos_src)*0.05)*2]))
first_dst = list(set(pos_dst[int(len(pos_dst)*0.05):int(len(pos_dst)*0.05)*2]))
second_dst = list(set(first_dst)-set(first_src))

for i in first_src:
    for j in second_dst:
        if (i,j) not in edge_all and (j,i) not in edge_all:
            if (j,i) not in us_edges and (j,i) not in us_edges:
                if i != j:
                    neg_src.append(i)
                    neg_dst.append(j)
                    neg_label.append(0)
print(len(neg_src))

pos_num = int(len(pos_src)*0.1)
# pos_num1 = int(len(pos_src)*0.2)
pos_num2 = int(len(pos_src))
pos_train_num = int(len(pos_src)*0.9)

src_train = pos_src[pos_num:pos_num2] + neg_src[pos_num:pos_num2] #第一列90%的节点
dst_train = pos_dst[pos_num:pos_num2] + neg_dst[pos_num:pos_num2] #第二列
label_train = pos_label[pos_num:pos_num2] + neg_label[pos_num:pos_num2]



with open("./划分数据/netscience_edge_train10.txt","w") as f:
    for i in range(pos_train_num+1):
        print(src_train[i],dst_train[i],file=f)
        
neg_node_train = neg_src[pos_num:pos_num2]+neg_dst[pos_num:pos_num2]
negtrain = pd.DataFrame()
negtrain['node'] = neg_node_train
negtrain.to_csv('./划分数据/netscience_neg_train10.csv',index = False)
usair = pd.DataFrame()
usair['src'] = src_train
usair['dst'] = dst_train
usair['label'] = label_train
usair.to_csv('./划分数据/netscience_usair_train10.csv',index = False)
###############################################################

src = pos_src[:pos_num] + neg_src[:pos_num]
dst = pos_dst[:pos_num] + neg_dst[:pos_num]
label = pos_label[:pos_num] + neg_label[:pos_num]

with open("./划分数据/netscience_edge_test10.txt",'w') as f:
    for i in range(pos_num):
        print(src[i], dst[i],file=f)

neg_node_test = pos_src[:pos_num]+neg_dst[:pos_num]
negtest = pd.DataFrame()
negtest['node'] = neg_node_test
negtest.to_csv('./划分数据/netscience_neg_test10.csv',index=False)
usair1 = pd.DataFrame()
usair1['src'] = src
usair1['dst'] = dst
usair1['label'] = label
usair1.to_csv('./划分数据/netscience_usair_test10.csv',index = False)
