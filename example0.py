#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:08:02 2022

@author: ozgesurer
"""

from scoreCART import scoreCART
from random import sample
import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter



def column(matrix, i):
    return [row[i] for row in matrix]

# inputs
max_depth = 3
min_node_size = 100
num_quantiles = 20
total_reps = 4 
alpha = .2
tol = 0
data_title = "airfoil"


def synthetic0(n):
    x = np.random.uniform(low=-1, high=1, size=n)
    data = []
    for id_x, x_i in enumerate(x):
        if x_i < 0:
            y = np.random.normal(loc=1, scale=2, size=1)
        else:
            y = np.random.exponential(scale=1.0, size=1)
        
        data.append([x_i, float(y)])
    return data

n = 1000
rows = synthetic0(n)
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
