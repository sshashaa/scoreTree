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

# Example: is1 is working best

shape1, scale1 = 7, 2  
s1 = np.random.gamma(shape1, scale1, 1000)

shape2, scale2 = 8, 3 
s2 = np.random.gamma(shape2, scale2, 1000)

shape3, scale3 = 9, 5
s3 = np.random.gamma(shape3, scale3, 1000)

shape4, scale4 = 1, 10
s4 = np.random.gamma(shape4, scale4, 1000)

plt.hist(s1, density=True, alpha=0.5, label='-1 < x < -0.5', bins=50, color='blue')
plt.hist(s2, density=True, alpha=0.5, label='-0.5 < x < 0', bins=50, color='red')
plt.hist(s3, density=True, alpha=0.5, label='0 < x < 0.5', bins=50, color='green')
plt.hist(s4, density=True, alpha=0.5, label='0.5 < x < 1', bins=50, color='orange')
plt.savefig('synth_fig/synth3_hist.png')
plt.legend()
plt.show()

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

filename = 'synth3_minnode_' + str(min_node_size) + '_n_' + str(n)


def synthetic1(n):
    x = np.random.uniform(low=-1, high=1, size=n)
    data = []
    for id_x, x_i in enumerate(x):
        if x_i < 0:
            if x_i < -0.5:
                y = np.random.gamma(shape1, scale1, 1)
            else:
                y = np.random.normal(shape2, scale2, 1)                    
        else:
            if x_i < 0.5:
                y = np.random.gamma(shape3, scale3, 1)
            else:
                y = np.random.normal(shape4, scale4, 1)                
        
        data.append([x_i, float(y)])
    return data

rows = synthetic1(n)
x_dim = len(rows[0])-1
#### #### #### #### #### #### #### ####

is_cat = []
cov_uniqvals = []
for i in range(x_dim):
    unique_vals = list(sorted(set(column(rows, i))))
    cov_uniqvals += [unique_vals]
    if len(unique_vals) <= 2:#len(rows)/len(unique_vals) > 100:
        is_cat += [1]
    else:
        is_cat += [0]


def OneRep(k):
    random.seed(k + 100)
    # Creates a training and a test set        
    holdout_size = int(len(rows)/2) 
    train_index = list(random.sample(range(len(rows)), holdout_size))
    test_index = list(set(range(len(rows))) - set(train_index))
    
    train_set = [rows[index] for index in train_index]    
    test_set = [rows[index] for index in test_index]
    dataset = [train_set, test_set]
    
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
    prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
    for pr in prune_thr_list:
        for m in methods: 
            # Fit the tree model
            CARTmodel = scoreCART(m, 
                                  train_set, 
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
                d = {'Method': m, 'Metric': metr, 'Train': np.round(dict_eval[metr][0], 2), 'Test': np.round(dict_eval[metr][1], 2), 'Threshold': pr}
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
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]

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
plt.savefig('synth_fig/test_' + filename + '.png')
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
plt.savefig('synth_fig/train_' + filename + '.png')
plt.show()

df_scores.to_csv('synth_fig/' + filename + '.csv', sep=',')