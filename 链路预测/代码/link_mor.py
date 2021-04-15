# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:41:58 2021

@author: 张玮
"""


#特征提取
import networkx as nx
import pandas as pd
import numpy as np
import random
import itertools
import matplotlib as plt

#对应模体M1
def three_one_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    num = len(u_mor) + len(v_mor)
    return num

#对应模体M2，寻找到两个节点共同邻居数
def three_two_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if (u_friends == [] or v_friends == []):
        return 0
    else:
        return len(set(u_friends) & set(v_friends))
    
#对应模体M3，思路：找到两个节点各自的邻居节点，让邻居节点各自两两组合，让其判断是否相连，根据M3模体，如果相连就系数deta1-1
def four_one_morphology(u,v,g):
    edge_all = list(g.edges())
    u_friends = list(g.neighbors(u)) #得到邻居节点
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    u_list = []
    v_list = []
    if len(u_mor) <=1:
        deta1 = 0
    else:
        for i in itertools.combinations(u_mor, 2):
            u_list.append(i)
        deta1 = int(len(u_list))
        for p,q in u_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta1 -= 1
            else:
                deta1 += 0
    if len(v_mor) <= 1:
        deta2 = 0
    else:
        for j in itertools.combinations(v_mor, 2):
            v_list.append(j)
        deta2 = int(len(v_list))
        for p,q in v_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta2 -= 1
            else:
                deta2 += 0
    return deta1+deta2
    
#M4
def four_two_morphology(u,v,g):
    edge_all = list(g.edges())
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    mor_list = []
    if (u_mor == []) or (v_mor == []):
        return 0
    else:
        for i in u_mor:
            for j in v_mor:
                mor_list.append((i,j))
        deta = int(len(mor_list))
        for p,q in mor_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta -= 1
        return deta
 
#M5    
def four_two2_morphology(u,v,g):
    edge_all = g.edges
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    edge_all = g.edges
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    detau = 0
    if (u_mor == []):
        detau = 0
    else:
        for i in u_mor:
            i_list = list(g.neighbors(i))
        if u in i_list:
            i_list.remove(u)
        if v in i_list:
            i_list.remove(v)
        for j in i_list:
            if (u,j) in edge_all or (j,u) in edge_all or (v,j) in edge_all or (j,v) in edge_all:
                detau += 1
    detav = 0
    if (v_mor == []):
        detav = 0
    else:
        for p in v_mor:
            p_list = list(g.neighbors(p))
            if u in p_list:
                p_list.remove(u)
            if v in p_list:
                p_list.remove(v)
            for q in p_list:
                if (u,q) in edge_all or (q,u) in edge_all or (v,q) in edge_all or (q,v) in edge_all:
                    detav += 1
    return detau+detav

#M6(计算为0)
def four_three_morphology(u,v,g):
    edge_all = list(g.edges())
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    u_list = []
    v_list = []
    if len(u_mor) <=1 :
        deta1 = 0
    else:
        for i in itertools.combinations(u_mor,2):
            u_list.append(i)
        deta1 = 0
        for p,q in u_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta1 += 1
            else:
                deta1 += 0
      
    if len(v_mor) <= 1:
        deta2 = 0
    else:
        for i in itertools.combinations(v_mor,2):
            v_list.append(i)
        deta2 = 0
        for p,q in v_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta2 += 1
            else:
                deta2 += 0
    return deta1+deta2
  
#M7          
def four_three2_morphology(u,v,g): 
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        deta = 0
        for i in cn:
            i_list = list(g.neighbors(i))
            i_list.remove(u)
            i_list.remove(v)
            if i_list == []:
                deta1 = 0
            else:
                deta1 = len(i_list)
                for j in i_list:
                    if (u,j) in edge_all or (j,u) in edge_all or (v,j) in edge_all or (j,v) in edge_all:
                        deta1 -= 1
            deta += deta1
        return deta

#M8    
def four_three3_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    u_list = []
    v_list = []
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        deta1 = 0
        if u_mor == []:
            deta1 = 0
        else:
            for i in cn:
                for j in u_mor:
                    u_list.append((i,j))
            deta1 = len(u_list)
            for p,q in u_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta1 -= 1
                else:
                    deta1 += 0
        deta2 = 0
        if v_mor == []:
            deta2 = 0
        else:
            for i in cn:
                for j in v_mor:
                    v_list.append((i,j))
            deta2 = len(v_list)
            for p,q in v_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta2 -= 1
                else:
                    deta2 += 0            
        return deta1+deta2

#M9        
def four_four_morphology(u,v,g):
    edge_all = list(g.edges)
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    mor_list = []
    if (u_mor == []) or (v_mor == []):
        return 0
    else:
        for i in u_mor:
            for j in v_mor:
                mor_list.append((i,j))
        deta = 0
        for p,q in mor_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta += 1
        return deta

#M10
def four_six1_morphology_else(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        if len(cn) <= 1:
            return 0
        else:
            cn_edge = []
            for i in itertools.combinations(cn, 2):
                cn_edge.append(i)
            d1 = 0
            for p,q in cn_edge:
                if (p,q) in edge_all or (q,p) in edge_all:
                    d1 += 1
                else:
                    d1 += 0
            deta = int(len(cn_edge)) - d1
            return deta

#M11                
def four_five2_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    u_list = []
    v_list = []
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        deta1 = 0
        if u_mor == []:
            deta1 = 0
        else:
            for i in cn:
                for j in u_mor:
                    u_list.append((i,j))
            for p,q in u_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta1 += 1
                else:
                    deta1 += 0
        deta2 = 0
        if v_mor == []:
            deta2 = 0
        else:
            for i in cn:
                for j in v_mor:
                    v_list.append((i,j))
            for p,q in v_list:
                if (p,q) in edge_all or (q,p) in edge_all:
                    deta2 += 1
                else:
                    deta2 += 0            
        return deta1+deta2

#M12                
def four_six2_morphology_else(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        cn = list(set(u_friends) & set(v_friends))
        if len(cn) <= 1:
            return 0
        else:
            cn_edge = []
            for i in itertools.combinations(cn,2):
                cn_edge.append(i)
            deta = 0
            for p,q in cn_edge:
               if (p,q) in edge_all or (q,p) in edge_all:
                    deta += 1
               else:
                   deta += 0
            return deta                

G = nx.Graph()
G.add_nodes_from(['a','b','c','d'])
G.add_edge('a','c')
G.add_edge('c','d')
G.add_edge('b','d')
# M1 = three_one_morphology('a', 'b', G)
# M2 = three_two_morphology('a', 'b', G)
# M3 = four_one_morphology('a', 'b', G)
# M4 = four_two_morphology('a', 'b', G)
# M5 = four_two2_morphology('a', 'b', G)
nx.draw(G,with_labels = True)
M6 = four_three_morphology('a', 'b', G)
print(M6)

negtest = pd.read_csv('./划分数据/netscience_neg_test10.csv') #10%的负样本的第一列节点和第二列节点，10%的负样本的所有负样本编号
neg_list = list(set(list(negtest['node']))) #set([iterable])函数创建一个无序不重复元素集，可进行关系测试，删除重复数据
g = nx.read_edgelist("./划分数据/netscience_edge_test10.txt",nodetype=int)
g = g.to_undirected()
for i in neg_list:
    if i in list(g.nodes()):
        pass
    else:
        g.add_node(i)
node_all = list(g.nodes())

negtrain = pd.read_csv('./划分数据/netscience_neg_train10.csv')
neg_list2 = list(set(list(negtrain['node'])))
G= nx.read_edgelist("./划分数据/netscience_edge_train10.txt",nodetype=int)
G = G.to_undirected()
for i in neg_list2+node_all:
    if i in list(G.nodes()):
        pass
    else:
        G.add_node(i)
edge_all = G.edges()

sf_links = pd.read_csv('./划分数据/netscience_usair_train10.csv')

sf_links['M1'] = sf_links[['src','dst']].apply(lambda r: three_one_morphology(r['src'], r['dst'],G), axis = 1)
sf_links['M2'] = sf_links[['src','dst']].apply(lambda r: three_two_morphology(r['src'], r['dst'],G),axis = 1)
sf_links['M3'] = sf_links[['src','dst']].apply(lambda r: four_one_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M4'] = sf_links[['src','dst']].apply(lambda r: four_two_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M5'] = sf_links[['src','dst']].apply(lambda r: four_two2_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M6'] = sf_links[['src','dst']].apply(lambda r: four_three_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M7'] = sf_links[['src','dst']].apply(lambda r: four_three2_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M8'] = sf_links[['src','dst']].apply(lambda r: four_three3_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M9'] = sf_links[['src','dst']].apply(lambda r: four_four_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M10'] = sf_links[['src','dst']].apply(lambda r: four_six1_morphology_else(r['src'], r['dst'], G),axis = 1)
sf_links['M11'] = sf_links[['src','dst']].apply(lambda r: four_five2_morphology(r['src'], r['dst'], G),axis = 1)
sf_links['M12'] = sf_links[['src','dst']].apply(lambda r: four_six2_morphology_else(r['src'], r['dst'], G),axis = 1)

sf_links.to_csv('netsciencew_linkmorj_train_final1.csv',index = None)


sf_links1 = pd.read_csv('./划分数据/netscience_usair_test10.csv')

sf_links1['M1'] = sf_links1[['src','dst']].apply(lambda r: thre  他 xdse_one_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M2'] = sf_links1[['src','dst']].apply(lambda r: three_two_morphology(r['src'], r['dst'], G),axis = 1) 
sf_links1['M3'] = sf_links1[['src','dst']].apply(lambda r: four_one_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M4'] = sf_links[['src','dst']].apply(lambda r: four_two_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M5'] = sf_links[['src','dst']].apply(lambda r: four_two2_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M6'] = sf_links[['src','dst']].apply(lambda r: four_three_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M7'] = sf_links[['src','dst']].apply(lambda r: four_three2_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M8'] = sf_links[['src','dst']].apply(lambda r: four_three3_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M9'] = sf_links[['src','dst']].apply(lambda r: four_four_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M10'] = sf_links[['src','dst']].apply(lambda r: four_six1_morphology_else(r['src'], r['dst'], G),axis = 1)
sf_links1['M11'] = sf_links[['src','dst']].apply(lambda r: four_five2_morphology(r['src'], r['dst'], G),axis = 1)
sf_links1['M12'] = sf_links[['src','dst']].apply(lambda r: four_six2_morphology_else(r['src'], r['dst'], G),axis = 1) 

sf_links1.to_csv('netsciencew_linkmorj_test_final1.csv',index = None)    
    
        
    