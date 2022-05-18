#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:43:15 2022

@author: sarashashaani
"""
import pandas as pd
import numpy as np
from scipy import stats

rows = pd.read_csv('synth_fig/df1.csv')

metrics = ['sse', 'crps', 'dss', 'is1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]

total_reps = 30
liste = []

for i in range(total_reps):
    ls = rows[i].copy()
    ls['Rep'] = i
    liste.append(ls)
df_scores = pd.concat(liste)


pval_test = np.zeros((4,4,5))
for th_id, th in enumerate(prune_thr_list):
    for metric_id, met in enumerate(metrics):
        for meth_id, meth in enumerate(metrics):
            df = df_scores[(df_scores['Method'] == meth) & (df_scores['Metric'] == met) & (df_scores['Threshold'] == th)]
            df_met = df_scores[(df_scores['Method'] == met) & (df_scores['Metric'] == met) & (df_scores['Threshold'] == th)]
            if meth_id == metric_id:
                pval_test[metric_id][meth_id][th_id] = 1
            else:
                pval_test[metric_id][meth_id][th_id] = stats.ttest_ind(df.Test, df_met.Test, equal_var=False).pvalue


pval_train = np.zeros((4,4,5))
for th_id, th in enumerate(prune_thr_list):
    for metric_id, met in enumerate(metrics):
        for meth_id, meth in enumerate(metrics):
            df = df_scores[(df_scores['Method'] == meth) & (df_scores['Metric'] == met) & (df_scores['Threshold'] == th)]
            df_met = df_scores[(df_scores['Method'] == met) & (df_scores['Metric'] == met) & (df_scores['Threshold'] == th)]
            if meth_id == metric_id:
                pval_train[metric_id][meth_id][th_id] = 1
            else:
                pval_train[metric_id][meth_id][th_id] = stats.ttest_ind(df.Train, df_met.Train, equal_var=False).pvalue
