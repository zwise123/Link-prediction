# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:13:46 2021

@author: 张玮
"""


import pandas as pd
import networkx as nx
import numpy as np
import random
import itertools
import math

  
def CAR(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = set(u_friends) & set(v_friends)
        deta = 0
        for s in cn:
            s_n = list(g.neighbors(s))
            s_n.remove(u)
            s_n.remove(v)
            s_set = []
            for i in s_n:
                if i in cn:
                    s_set.append(i)
            num_s = len(s_set)/2 
            deta += num_s
        cn_num = len(cn)
        car = cn_num*deta
        return car
        

num_list = []
for i in range(10):
    num_list.append(i)

for qq in num_list:            
    negtest = pd.read_csv('hii05_neg_test1'+str(qq)+'.csv')  #10%的负样本的第一列节点和第二列节点，10%的负样本的所有负样本编号
    neg_list = list(set(list(negtest['node'])))  #set([iterable])函数创建一个无序不重复元素集，可进行关系测试，删除重复数据
    g= nx.read_edgelist("hii05_edge_test1"+str(qq)+".txt",nodetype=int)
    g = g.to_undirected()
    for i in neg_list:
        if i in list(g.nodes()):
            pass
        else:
            g.add_node(i)
    node_all = list(g.nodes())

    negtrain = pd.read_csv('hii05_neg_train1'+str(qq)+'.csv')
    neg_list = list(set(list(negtrain['node'])))
    G= nx.read_edgelist("hii05_edge_train1"+str(qq)+".txt",nodetype=int)
    G = G.to_undirected()
    for i in neg_list+node_all:
        if i in list(G.nodes()):
            pass
        else:
            G.add_node(i)
    edge_all = G.edges()

    
    sf_links1 = pd.read_csv('hii05_usair_test1'+str(qq)+'.csv')

    sf_links1['CAR'] = sf_links1[['src','dst']].apply(lambda r: CAR(r['src'], r['dst'],G), axis =1)
   
#sf_links.to_csv('T131_mor_train_final.csv',index = None)
    sf_links1.to_csv('hii05_CAR_test_final1'+str(qq)+'.csv',index = None)
    
    
    
    
    print ('iter:',qq)
