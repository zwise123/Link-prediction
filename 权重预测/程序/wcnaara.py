# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:20:36 2019

@author: Administrator
"""

import pandas as pd
import networkx as nx
#import igraph as ig
import numpy as np
import math
#import random


def wcommon_friends(u, v, g):
    u_friends = g.neighbors(u)
    v_friends = g.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn =  set(u_friends) & set(v_friends)
        wcn = 0
        for i in cn:
            wu = g[u][i]['weight']
            wv = g[i][v]['weight']
            w = (wu+wv)/2
            wcn += w
        return wcn

def WAA(u,v,g):
    u_friends = g.neighbors(u)
    v_friends = g.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = set(u_friends) & set(v_friends)
        waa = 0
        for i in cn:
            wu = g[u][i]['weight']
            wv = g[i][v]['weight']
            w = wu+wv
            i_friends = g.neighbors(i)
            strength = 0
            for j in i_friends:
                s= g[i][j]['weight']
                strength += s
            if strength == 0:
                waa += 0
            else:
                waa += w/(2*(math.log(1+strength)))
        return waa
    
def WRA(u,v,g):
    u_friends = g.neighbors(u)
    v_friends = g.neighbors(v)
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = set(u_friends) & set(v_friends)
        wra = 0
        for i in cn:
            wu = g[u][i]['weight']
            wv = g[i][v]['weight']
            w = wu+wv
            i_friends = g.neighbors(i)
            strength = 0
            for j in i_friends:
                s= g[i][j]['weight']
                strength += s
            if strength == 0:
                wra += 0
            else:
                wra += w/float(2*strength)
        return wra

num_list = []
for i in range(10):
    num_list.append(i)

for qq in num_list:
    G = nx.read_edgelist('geom_weight.txt', nodetype=int, data=(('weight',float),))
    G1 = nx.read_edgelist('geom_test1'+str(qq)+'.txt', nodetype=int, data=(('weight',float),))
    for edge in G1.edges():
        G[edge[0]][edge[1]]['weight'] = 0
    
#g= nx.read_edgelist("condmat_edge_part.txt",nodetype=int)
    G = G.to_undirected()

#sf_links = pd.read_csv('graph_train_random.csv')
    sf_links = pd.read_csv('geom_test1'+str(qq)+'.csv')
    sf_links['WAA'] = sf_links[['src','dst']].apply(lambda r: WAA(r['src'], r['dst'],G), axis =1)
    sf_links['WRA'] = sf_links[['src','dst']].apply(lambda r: WRA(r['src'], r['dst'],G), axis =1)
    sf_links['WCN'] = sf_links[['src','dst']].apply(lambda r: wcommon_friends(r['src'], r['dst'],G), axis =1)
    sf_links.to_csv('geomw_wcnaara_test_final1'+str(qq)+'.csv',index = None)









