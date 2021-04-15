# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:54:11 2019

@author: Administrator
"""

import pandas as pd
import networkx as nx
#import igraph as ig
import numpy as np
import random  
import math

 
def rWCN(u, v, g):
    u_friends = g.neighbors(u)
    v_friends = g.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = set(u_friends) & set(v_friends)
        rwcn = 0
        for i in cn:
            Wui = g[u][i]['weight']
            Wiv = g[i][v]['weight']
            w = Wui*Wiv
            rwcn =+ w
        return rwcn
        
def rWAA(u, v, g):
    u_friends = g.neighbors(u)
    v_friends = g.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = set(u_friends) & set(v_friends)
        rwaa = 0
        for i in cn:
            wui = g[u][i]['weight']
            wiv = g[i][v]['weight']
            w = wui*wiv
            i_friends = g.neighbors(i)
            sumweight = 0
            for j in i_friends:
                ws = g[i][j]['weight']
                sumweight += ws
            if sumweight == 0 or math.log(1+sumweight) ==0:
                rwaa += 0
            else:
                rwaa += w/math.log(1+sumweight)
        return rwaa
    
def rWRA(u, v, g):
    u_friends = g.neighbors(u)
    v_friends = g.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = set(u_friends) & set(v_friends)
        rwra = 0
        for i in cn:
            wui = g[u][i]['weight']
            wiv = g[i][v]['weight']
            w = wui*wiv
            i_friends = g.neighbors(i)
            sumweight = 0
            for j in i_friends:
                ws = g[i][j]['weight']
                sumweight += ws
            if sumweight == 0:
                rwra += 0
            else:
                rwra += w/float(sumweight)
        return rwra
 
        
num_list = []
for i in range(10):
    num_list.append(i)

# for qq in num_list:        
G = nx.read_edgelist('权重预测/数据/权重归一化后数据/USAir.txt', nodetype=int, data=(('weight',float),))
G1 = nx.read_edgelist('usa_test_random.csv', nodetype=int, data=(('weight',float),))
for edge in G1.edges():
    G[edge[0]][edge[1]]['weight'] = 0

G = G.to_undirected()

sf_links = pd.read_csv('usa_test_random.csv')

sf_links['rWCN'] = sf_links[['src','dst']].apply(lambda r: rWCN(r['src'], r['dst'],G), axis =1)
sf_links['rWAA'] = sf_links[['src','dst']].apply(lambda r: rWAA(r['src'], r['dst'],G), axis =1)
sf_links['rWRA'] = sf_links[['src','dst']].apply(lambda r: rWRA(r['src'], r['dst'],G), axis =1)

sf_links.to_csv('usa_rw_test_final1.csv',index = None)


sf_links = pd.read_csv('usa_train_random.csv')

sf_links['rWCN'] = sf_links[['src','dst']].apply(lambda r: rWCN(r['src'], r['dst'],G), axis =1)
sf_links['rWAA'] = sf_links[['src','dst']].apply(lambda r: rWAA(r['src'], r['dst'],G), axis =1)
sf_links['rWRA'] = sf_links[['src','dst']].apply(lambda r: rWRA(r['src'], r['dst'],G), axis =1)

sf_links.to_csv('usa_rw_train_final1.csv',index = None)