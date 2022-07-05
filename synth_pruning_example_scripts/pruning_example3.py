#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 14:06:28 2022

@author: ozgesurer
"""


from scoreCARTprune import scoreCART
from random import sample
import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter
import pandas as pd

# Example: crps and sse are not doing well. dss is doing well
np.random.seed(12345)

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
plt.legend()
plt.show()

def column(matrix, i):
    return [row[i] for row in matrix]

# inputs
max_depth = 4
min_node_size = 100
num_quantiles = 20
total_reps = 4 
alpha = .2
tol = 0


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

n = 5000
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

# Creates a training and a test set        
holdout_size = int(len(rows)/2) 
train_index = list(sample(range(len(rows)), holdout_size))
test_index = list(set(range(len(rows))) - set(train_index))

train_set = [rows[index] for index in train_index]    
test_set = [rows[index] for index in test_index]
dataset = [train_set, test_set]
args = {}

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
#prune_thr_list = [0.8]

for m in methods: 
    if m in ["crps", "dss"]:
        pr = 0.3
    else:
        pr = 0.1
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
    fittedtree = CARTmodel.fittedtree
    
    dict_eval = CARTmodel.accuracy_val(test_set, 
                                       actual, 
                                       self_test, 
                                       actual_in, 
                                       metrics=['sse', 'crps', 'dss', 'is1'])
    for metr in ['sse', 'crps', 'dss', 'is1']:
        d = {'Method': m, 'Metric': metr, 'Train': np.round(dict_eval[metr][0], 2), 'Test': np.round(dict_eval[metr][1], 2)}
        dictable.append(d)
dfres = pd.DataFrame(dictable)
print(dfres)    

for metr in ['sse', 'crps', 'dss', 'is1']:
    trerror = dfres[dfres['Metric'] == metr]['Train']
    plt.plot(np.arange(len(trerror)), trerror)
    plt.ylabel(metr)
    plt.title('Training')
    plt.show()
    
    trerror = dfres[dfres['Metric'] == metr]['Test']
    plt.plot(np.arange(len(trerror)), trerror)
    plt.ylabel(metr)
    plt.title('Test')
    plt.show()