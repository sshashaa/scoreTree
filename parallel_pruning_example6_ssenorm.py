#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:06:28 2022

@author: ozgesurer
"""

from scoreCARTprune import scoreCART
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import Parallel, delayed
import random
import seaborn as sns


def column(matrix, i):
    return [row[i] for row in matrix]

# inputs
max_depth = 4
min_node_size = 10
num_quantiles = 20
total_reps = 30 
alpha = .2
tol = 0
n = 1600

filename = 'synth6' + '_n_' + str(n)

def OneRep(k):
    # random.seed(k + 10)
    filename = 'less_noise_examples/synth6_rep_' + str(k) + '.csv'
    rows = pd.read_csv(filename, header=None)
    rows = rows.values.tolist()[0:n]
    is_cat = []
    cov_uniqvals = []
    x_dim = len(rows[0])-1
    for i in range(x_dim):
        unique_vals = list(sorted(set(column(rows, i))))
        cov_uniqvals += [unique_vals]
        if len(unique_vals) <= 2:
            is_cat += [1]
        else:
            is_cat += [0]

    filename = 'less_noise_examples/synth6_test' + '.csv'
    rows_test = pd.read_csv(filename, header=None)
    rows_test = rows_test.values.tolist()

    dataset = [rows, rows_test]
    
    test_set = list()
    for row in dataset[1]:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
    
    self_test = list()
    for row in dataset[0]:
        row_copy = list(row)
        self_test.append(row_copy)
        row_copy[-1] = None
    
    actual = [row[-1] for row in dataset[1]]
    actual_in = [row[-1] for row in dataset[0]]
    
    methods = ["crps", "dss", "is1", "sse"]
    dictable = []
    prune_thr_list = [0, 0.1]
    for pr in prune_thr_list:
        for m in methods: 
            # Fit the tree model
            CARTmodel = scoreCART(m, 
                                  rows, 
                                  tol, 
                                  max_depth, 
                                  min_node_size, 
                                  num_quantiles, 
                                  alpha,
                                  pr,
                                  args = {'is_cat': is_cat, 'cov_uniqvals': cov_uniqvals})
            CARTmodel.build_tree()
            # fittedtree = CARTmodel.fittedtree
            dict_eval = CARTmodel.accuracy_val(test_set, 
                                               actual, 
                                               self_test, 
                                               actual_in, 
                                               metrics=['sse', 'crps', 'dss', 'is1'])
            for metr in ['sse', 'crps', 'dss', 'is1']:
                d = {'Method': m, 'Metric': metr, 'Train': np.round(dict_eval[metr][1], 2), 'Test': np.round(dict_eval[metr][0], 2), 'Threshold': pr}
                dictable.append(d)
    dfres = pd.DataFrame(dictable)
    return dfres

# Run parallel for each replicate
scores_reps = Parallel(n_jobs=min(total_reps, 20))(delayed(OneRep)(rep_no) for rep_no in range(total_reps))   


liste = []
for i in range(total_reps):
    ls = scores_reps[i].copy()
    ls['Rep'] = i
    liste.append(ls)
df_scores = pd.concat(liste)


metrics = ['sse', 'crps', 'dss', 'is1']
prune_thr_list = [0, 0.1]

fig, axes = plt.subplots(len(metrics), len(prune_thr_list), sharex=True, figsize=(25, 25))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font) 
for metric_id, met in enumerate(metrics):
    for th_id, th in enumerate(prune_thr_list):
        df = df_scores[(df_scores['Metric'] == met) & (df_scores['Threshold'] == th)]
        sns.boxplot(ax=axes[metric_id, th_id], x="Method", y="Test", data=df,  palette="Set3")
        axes[metric_id, th_id].set_ylabel('')
        if th == 0:
            axes[metric_id, th_id].set_ylabel('Test' + ' (' + met + ')')
        if metric_id == 0:
            axes[metric_id, th_id].set_title('Threshold: ' + str(th))
plt.savefig('less_noise_examples/test_' + filename + '.png')
plt.show()
        
fig, axes = plt.subplots(len(metrics), len(prune_thr_list), sharex=True, figsize=(25, 25))

for metric_id, met in enumerate(metrics):
    for th_id, th in enumerate(prune_thr_list):
        df = df_scores[(df_scores['Metric'] == met) & (df_scores['Threshold'] == th)]
        sns.boxplot(ax=axes[metric_id, th_id], x="Method", y="Train", data=df,  palette="Set3")
        axes[metric_id, th_id].set_ylabel('')
        if th == 0:
            axes[metric_id, th_id].set_ylabel('Train' + ' (' + met + ')')
        if metric_id == 0:
            axes[metric_id, th_id].set_title('Threshold: ' + str(th))
plt.savefig('less_noise_examples/train_' + filename + '.png')
plt.show()

df_scores.to_csv('less_noise_examples/' + filename + '.csv', sep=',')