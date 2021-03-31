# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:06:45 2021

@author: 张玮
"""


#相关性分析
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame()
df = pd.read_csv('链路预测/代码/netsciencew_linkmorj_train_final1.csv')
fig,ax = plt.subplots(figsize=(20,15))
annot_kws = {'size':6}
sns.set(font_scale=1.5)


#net train linkmorj
sns.heatmap(df[['M1','M3','M9','M5','M4','M7','M2','M8','M12','M6','M10','M11']].corr(method='pearson'),
                     annot=False,vmin=0,cmap='Blues',yticklabels=['M1','M3','M9','M5','M4','M7','M2','M8','M12','M6','M10','M11']
                    ,xticklabels=['M1','M3','M9','M5','M4','M7','M2','M8','M12','M6','M10','M11'])
ax.broken_barh([(0,5)],(0,5),facecolors=('none'),edgecolors=('black'),linewidth=5)
ax.broken_barh([(5,7)], (5,7),facecolors=('none'),edgecolors=('black'),linewidth=5)
#ax.broken_barh([(10,1)], (10,1),facecolors=('none'),edgecolors=('black'),linewidth=5)
#ax.broken_barh([(11,1)], (11,1),facecolors=('none'),edgecolors=('black'),linewidth=5)

plt.yticks(rotation = 360,fontsize=24) 
plt.xticks(rotation = 360,fontsize=24)
plt.savefig('netlink16.pdf')
plt.show()