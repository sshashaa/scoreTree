#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:26:16 2022

@author: ozgesurer
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import seaborn as sns

n = [200, 400]
methods = ['crps', 'dss', 'is1']
prune_thr_list = [0, 0.1]
synth_id = [6, 7]


for ns in n:
    for thr in prune_thr_list:
        df_all = pd.DataFrame()
        
        for m in methods:
            for i in synth_id:
                if i == 6:
                    type_data = 'Easy'
                    
                    df_compact = pd.DataFrame()
                    filename = 'synth_fig/synth' + str(i) + '_minnode_10_n_' + str(ns) + '.csv'
                        
                    # Read csv.
                    df_scores = pd.read_csv(filename)
                    
                    df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
                    df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == thr)]
                    
                    data = {'Train': np.array(df1['Train']) -  np.array(df2['Train']),
                            'Test': np.array(df1['Test']) -  np.array(df2['Test']),
                            'Type': type_data,
                            'Method': m}
                
                    df6 = pd.DataFrame(data)
                elif i == 7:
                    type_data = 'Hard'
                    
                    df_compact = pd.DataFrame()
                    filename = 'synth_fig/synth' + str(i) + '_minnode_10_n_' + str(ns) + '.csv'
                        
                    # Read csv.
                    df_scores = pd.read_csv(filename)
                    
                    df1 = df_scores[(df_scores['Method'] == 'crps') & (df_scores['Metric'] == 'crps') & (df_scores['Threshold'] == thr)]
                    df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == 'crps') & (df_scores['Threshold'] == thr)]
                    
                    data = {'Train': np.array(df1['Train']) -  np.array(df2['Train']),
                            'Test': np.array(df1['Test']) -  np.array(df2['Test']),
                            'Type': type_data,
                            'Method': m}
                
                    df7 = pd.DataFrame(data)
            
            dfnew = df6.append(df7)
            df_all = df_all.append(dfnew)
     
        boxplot = sns.catplot(x="Type", y="Test",
                    col="Method",
                    data=df_all, kind="box", sharey=False)

        boxplot.fig.suptitle('n=' + str(ns) + ', thr=' + str(thr), fontsize=12)        
        boxplot = sns.catplot(x="Type", y="Train",
                    col="Method",
                    data=df_all, kind="box", sharey=False)
        boxplot.fig.suptitle('n=' + str(ns) + ', thr=' + str(thr), fontsize=12)   

           

        
            
            
            
            
            
            