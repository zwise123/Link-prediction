# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:33:45 2019

@author: Administrator
"""

import pandas as pd
import networkx as nx
import numpy as np
import random
import itertools

    
def three_one_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    u_friends.remove(v)
    v_friends.remove(u)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    num = len(u_mor) + len(v_mor)
    return num

def three_two_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if (u_friends == []) or (v_friends == []):
        return 0
    else:
        return len(set(u_friends) & set(v_friends)) 
    
def four_one_morphology(u,v,g):
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
    if len(u_mor) <= 1:
        deta1 = 0
    else:
        for i in itertools.combinations(u_mor,2):
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
        for j in itertools.combinations(v_mor,2):
            v_list.append(j)
        deta2 = int(len(v_list))
        for p,q in v_list:
            if (p,q) in edge_all or (q,p) in edge_all:
                deta2 -= 1
            else:
                deta2 += 0
    return deta1+deta2

def four_two_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    u_friends.remove(v)
    v_friends.remove(u)
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

def four_two2_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    if u in v_friends:
        v_friends.remove(u)
    if v in u_friends:
        u_friends.remove(v)
    u_mor = list(set(u_friends)-set(v_friends))
    v_mor = list(set(v_friends)-set(u_friends))
    detau = 0
    if (u_mor == []) :
        detau = 0
    else:
        for i in u_mor:
            i_list = list(g.neighbors(i))
            if u in i_list:
                i_list.remove(u)
            if v in i_list:
                i_list.remove(v)
            deta1 = len(i_list)
            for j in i_list:
                if (u,j) in edge_all or (j,u) in edge_all or (v,j) in edge_all or (j,v) in edge_all:
                    deta1 -= 1
            detau += deta1
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
            deta2 = len(p_list)
            for q in p_list:
                if (u,q) in edge_all or (q,u) in edge_all or (v,q) in edge_all or (q,v) in edge_all:
                    deta2 -= 1
            detav += deta2
    return detau+detav
        
def four_three_morphology(u,v,g):
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
    
def four_four_morphology(u,v,g):
    u_friends = list(g.neighbors(u))
    v_friends = list(g.neighbors(v))
    u_friends.remove(v)
    v_friends.remove(u)
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

def four_six1_morphology(u,v,g):
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
            d1 = 0
            for p,q in cn_edge:
               if (p,q) in edge_all or (q,p) in edge_all:
                   d1 += 1
               else:
                   d1 += 0
            deta = int(len(cn_edge)) - d1
            return deta

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
            
def four_six2_morphology(u,v,g):
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



G = nx.read_edgelist('权重预测/数据/权重归一化后数据/geom_weight.txt', nodetype=int, data=(('weight',float),))
G1 = nx.read_edgelist('geom_test_random.csv', nodetype=int, data=(('weight',float),))
for edge in G1.edges():
    G[edge[0]][edge[1]]['weight'] = 0
    
G = G.to_undirected()
edge_all = G.edges()

sf_links = pd.read_csv('geom_test_random.csv')

sf_links['M1'] = sf_links[['src','dst']].apply(lambda r: three_one_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M2'] = sf_links[['src','dst']].apply(lambda r: three_two_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M3'] = sf_links[['src','dst']].apply(lambda r: four_one_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M4'] = sf_links[['src','dst']].apply(lambda r: four_two_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M5'] = sf_links[['src','dst']].apply(lambda r: four_three_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M6'] = sf_links[['src','dst']].apply(lambda r: four_four_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M7'] = sf_links[['src','dst']].apply(lambda r: four_six1_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M8'] = sf_links[['src','dst']].apply(lambda r: four_six2_morphology(r['src'], r['dst'],G), axis =1)

sf_links['M9'] = sf_links[['src','dst']].apply(lambda r: four_two2_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M10'] = sf_links[['src','dst']].apply(lambda r: four_three2_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M11'] = sf_links[['src','dst']].apply(lambda r: four_three3_morphology(r['src'], r['dst'],G), axis =1)
sf_links['M12'] = sf_links[['src','dst']].apply(lambda r: four_five2_morphology(r['src'], r['dst'],G), axis =1)

sf_links.to_csv('geomw_4morj_test_final1.csv',index = None)
    
sf_links1 = pd.read_csv('geom_train_random.csv')

sf_links1['M1'] = sf_links1[['src','dst']].apply(lambda r: three_one_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M2'] = sf_links1[['src','dst']].apply(lambda r: three_two_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M3'] = sf_links1[['src','dst']].apply(lambda r: four_one_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M4'] = sf_links1[['src','dst']].apply(lambda r: four_two_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M5'] = sf_links1[['src','dst']].apply(lambda r: four_three_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M6'] = sf_links1[['src','dst']].apply(lambda r: four_four_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M7'] = sf_links1[['src','dst']].apply(lambda r: four_six1_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M8'] = sf_links1[['src','dst']].apply(lambda r: four_six2_morphology(r['src'], r['dst'],G), axis =1)

sf_links1['M9'] = sf_links1[['src','dst']].apply(lambda r: four_two2_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M10'] = sf_links1[['src','dst']].apply(lambda r: four_three2_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M11'] = sf_links1[['src','dst']].apply(lambda r: four_three3_morphology(r['src'], r['dst'],G), axis =1)
sf_links1['M12'] = sf_links1[['src','dst']].apply(lambda r: four_five2_morphology(r['src'], r['dst'],G), axis =1)

sf_links1.to_csv('geomw_4morj_train_final1.csv',index = None)