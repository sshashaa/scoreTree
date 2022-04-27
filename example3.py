#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:09:23 2022

@author: ozgesurer
"""

from scoreCART import scoreCART
from random import sample
import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter


shape, scale = 1., 1.  # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)
plt.hist(s, density=True, alpha=0.5, label='N1', bins=50, color='blue')


shape, scale = 0.1, 10  # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)
plt.hist(s, density=True, alpha=0.5, label='N2', bins=50, color='red')
plt.legend()
plt.show()

shape, scale = 10., 1 # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)
plt.hist(s, density=True, alpha=0.5, label='N3', bins=50, color='green')

shape, scale = 1., 10 # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)
plt.hist(s, density=True, alpha=0.5, label='N4', bins=50, color='orange')
plt.legend()
plt.show()


def column(matrix, i):
    return [row[i] for row in matrix]

# inputs
max_depth = 2
min_node_size = 100
num_quantiles = 20
total_reps = 4 
alpha = .2
tol = 0
data_title = "airfoil"


def synthetic1(n):
    x = np.random.uniform(low=-1, high=1, size=n)
    data = []
    for id_x, x_i in enumerate(x):
        if x_i < 0:
            if x_i < -0.5:
                shape, scale = 1., 1
                y = np.random.gamma(shape, scale, 1)
            else:
                shape, scale = 0.1, 10 
                y = np.random.gamma(shape, scale, 1)                    
        else:
            if x_i < 0.5:
                shape, scale = 10., 1
                y = np.random.gamma(shape, scale, 1)
            else:
                shape, scale = 1., 10
                y = np.random.gamma(shape, scale, 1)                
        
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

for m in methods: 
    # Fit the tree model
    CARTmodel = scoreCART(m, 
                          train_set, 
                          tol, 
                          max_depth, 
                          min_node_size, 
                          num_quantiles, 
                          alpha,
                          args = {'is_cat': is_cat, 'cov_uniqvals': cov_uniqvals})
    CARTmodel.build_tree()
    fittedtree = CARTmodel.fittedtree
    
    dict_eval = CARTmodel.accuracy_val(test_set, 
                                       actual, 
                                       self_test, 
                                       actual_in, 
                                       metrics=['sse', 'crps', 'dss', 'is1'])
