# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:48:50 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df=pd.DataFrame()
df=pd.read_csv('geomw_4morj_train_final10.csv')
fig, ax = plt.subplots(figsize=(20,15))
annot_kws={'size':6}
sns.set(font_scale=1.5)
     
sns.heatmap(df[['three_one','four_one','four_three1','four_two2','four_three3','four_five2','four_two1','four_four','four_five1','four_three2','three_two','four_six']].corr(method='pearson'),
                     annot=False,vmin=0,cmap='Blues',yticklabels=['3_1','4_1','4_3_1','4_2_2','4_3_3','4_5_2','4_2_1','4_4','4_5_1','4_3_2','3_2','4_6']
                    ,xticklabels=['3_1','4_1','4_3_1','4_2_2','4_3_3','4_5_2','4_2_1','4_4','4_5_1','4_3_2','3_2','4_6'])

ax.broken_barh([(0,7)], (0,7),facecolors=('none'),edgecolors=('black'),linewidth=5)
ax.broken_barh([(7,1)], (7,1),facecolors=('none'),edgecolors=('black'),linewidth=6)
ax.broken_barh([(8,1)], (8,1),facecolors=('none'),edgecolors=('black'),linewidth=6)
ax.broken_barh([(9,1)], (9,1),facecolors=('none'),edgecolors=('black'),linewidth=5)
ax.broken_barh([(10,2)], (10,2),facecolors=('none'),edgecolors=('black'),linewidth=5)
        
        
plt.yticks(rotation = 360) 
plt.xticks(rotation=360)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('geom_weight10.pdf')
plt.show()