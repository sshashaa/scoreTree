#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:22:35 2022

@author: ozgesurer
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import seaborn as sns
import matplotlib.pyplot as plt

n = [200, 400, 800, 1600]
methods = ['crps', 'dss', 'is1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
i = 6

# y-axis: Threshold
for ns in n:
    df_all = pd.DataFrame()
    filename = 'less_noise_examples/synth' + str(i) + '_n_' + str(ns) + '.csv'
    df_scores = pd.read_csv(filename)
    for thr in prune_thr_list:
        for m in methods:

            df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
            df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
            
            diff_train = np.array(df1['Train']) -  np.array(df2['Train'])
            diff_test = np.array(df1['Test']) -  np.array(df2['Test'])

            data = {'Train': diff_train/np.sqrt(np.var(diff_train)),
                    'Test': diff_test/np.sqrt(np.var(diff_test)),
                    'Method': m,
                    'Threshold': thr}
        
            df = pd.DataFrame(data)
            df_all = df_all.append(df)
    
    boxplot = sns.catplot(x="Threshold",
                          y="Train",
                          col="Method", 
                          data=df_all, 
                          kind="box", 
                          sharey=True)
    ax1, ax2, ax3 = boxplot.axes[0]
    ax1.axhline(0, ls='--', color='red')
    ax2.axhline(0, ls='--', color='red')
    ax3.axhline(0, ls='--', color='red')
    plt.savefig('paper_figures/train_' + 'synth_' + str(i) + '_n=' + str(ns) + '.png')     
    boxplot.fig.suptitle('n=' + str(ns), fontsize=14,  y=1.12)    
    
    boxplot = sns.catplot(x="Threshold", 
                          y="Test",
                          col="Method",
                          data=df_all, 
                          kind="box", 
                          sharey=True)
    ax1, ax2, ax3 = boxplot.axes[0]
    ax1.axhline(0, ls='--', color='red')
    ax2.axhline(0, ls='--', color='red')
    ax3.axhline(0, ls='--', color='red')
    plt.savefig('paper_figures/test_' + 'synth_' + str(i) + '_n=' + str(ns) + '.png') 
    boxplot.fig.suptitle('n=' + str(ns), fontsize=14, y=1.12)   


# y-axis: Sample size
for thr in prune_thr_list:
    df_all = pd.DataFrame()

    for ns in n:
        filename = 'less_noise_examples/synth' + str(i) + '_n_' + str(ns) + '.csv'
        df_scores = pd.read_csv(filename)
        for m in methods:

            df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
            df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
            
            diff_train = np.array(df1['Train']) - np.array(df2['Train'])
            diff_test = np.array(df1['Test']) - np.array(df2['Test'])

            data = {'Train': diff_train/np.sqrt(np.var(diff_train)),
                    'Test': diff_test/np.sqrt(np.var(diff_test)),
                    'Method': m,
                    'n': ns}
        
            df = pd.DataFrame(data)
            df_all = df_all.append(df)
     
    boxplot = sns.catplot(x="n",
                          y="Train",
                          col="Method", 
                          data=df_all, 
                          kind="box", 
                          sharey=True)
    ax1, ax2, ax3 = boxplot.axes[0]
    ax1.axhline(0, ls='--', color='red')
    ax2.axhline(0, ls='--', color='red')
    ax3.axhline(0, ls='--', color='red')
    plt.savefig('paper_figures/train_' + 'synth_' + str(i) + '_Threshold=' + str(thr) + '.png')   
    boxplot.fig.suptitle('Threshold=' + str(thr), fontsize=14,  y=1.12)    
    
    boxplot = sns.catplot(x="n", 
                          y="Test",
                          col="Method",
                          data=df_all, 
                          kind="box", 
                          sharey=True)
    ax1, ax2, ax3 = boxplot.axes[0]
    ax1.axhline(0, ls='--', color='red')
    ax2.axhline(0, ls='--', color='red')
    ax3.axhline(0, ls='--', color='red')
    plt.savefig('paper_figures/test_' + 'synth_' + str(i) + '_Threshold=' + str(thr) + '.png')   
    boxplot.fig.suptitle('Threshold=' + str(thr), fontsize=14, y=1.12)              

        
# y-axis: percentage
df_all = []
for ns in n:
    filename = 'less_noise_examples/synth' + str(i) + '_n_' + str(ns) + '.csv'
    df_scores = pd.read_csv(filename)
    for thr in prune_thr_list:
        for m in methods:

            df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
            df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
            
            diff_train = np.array(df1['Train']) > np.array(df2['Train'])
            diff_test = np.array(df1['Test']) > np.array(df2['Test'])

            data = {'Train(%)': 100*np.mean(diff_train),
                    'Test(%)': 100*np.mean(diff_test),
                    'Method': m,
                    'Threshold': thr,
                    'n': ns}

            df_all.append(data)
            

df_all = pd.DataFrame(df_all)

g = sns.FacetGrid(df_all, col='n', hue='Method')
g = g.map(sns.lineplot, 'Threshold', 'Train(%)', ci=None).add_legend()
plt.savefig('paper_figures/train_line_' + 'synth_' + str(i) + '_n' + '.png') 
            
g = sns.FacetGrid(df_all, col='Threshold', hue='Method')
g = g.map(sns.lineplot, 'n', 'Train(%)', ci=None).add_legend()    
plt.savefig('paper_figures/train_line_' + 'synth_' + str(i) + '_threshold' + '.png') 

g = sns.FacetGrid(df_all, col='n', hue='Method')
g = g.map(sns.lineplot, 'Threshold', 'Test(%)', ci=None).add_legend()
plt.savefig('paper_figures/test_line_' + 'synth_' + str(i) + '_n' + '.png') 
            
g = sns.FacetGrid(df_all, col='Threshold', hue='Method')
g = g.map(sns.lineplot, 'n', 'Test(%)', ci=None).add_legend()    
plt.savefig('paper_figures/test_line_' + 'synth_' + str(i) + '_threshold' + '.png')         

df_all.to_csv('paper_figures/' + 'anova_synth_' + str(i) + '.csv', sep=',')      
            
            