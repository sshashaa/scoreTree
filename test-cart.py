#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:52:30 2023

@author: sarashashaani

test CART
"""
from scoreCARTprune import scoreCART
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import Parallel, delayed
import random
import seaborn as sns
import random

import ast
import numpy as np
from collections import Counter

def column(matrix, i):
    return [row[i] for row in matrix]

# directory = "/home/sshasha2/"
data_title = 'yield'
datafile = "data/test_"+data_title+".txt"
# log_file = open(directory+"log_"+data_title+"_"+params+".txt", 'a+')
        
with open (datafile, 'r') as f: # use with to open your files, it close them automatically
    rows = [x.split() for x in f]    
rows = rows[1:]
for i in range(len(rows)):
    rows[i] = [float(x) for x in rows[i]]


test_size = 40
random.seed(0)       
test_index = list(random.sample(range(len(rows)), test_size))
rem_index = list(set(range(len(rows))) - set(test_index))
rem_set = [rows[index] for index in rem_index]    
test_set_fix = [rows[index] for index in test_index]

# inputs
max_depth = 3
min_node_size = 10
num_quantiles = 20
total_reps = 1#30 
alpha = .2
tol = 0
n = 40

methods = ["crps", "dss", "is1", "sse"]
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]

tr_set = random.sample(rem_set, n)
dataset = [tr_set, test_set_fix]

is_cat = []
cov_uniqvals = []
x_dim = len(tr_set[0])-1
for i in range(x_dim):
    unique_vals = list(sorted(set(column(tr_set, i))))
    cov_uniqvals += [unique_vals]
    if len(unique_vals) <= 2:
        is_cat += [1]
    else:
        is_cat += [0]

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

dictable = []
pr = 0.1
m = 'sse'

# Fit the tree model
CARTmodel = scoreCART(m, 
                      tr_set, 
                      tol, 
                      max_depth, 
                      min_node_size, 
                      num_quantiles, 
                      alpha,
                      pr,
                      args = {'is_cat': is_cat, 'cov_uniqvals': cov_uniqvals})
CARTmodel.build_tree()


# dict_eval = CARTmodel.accuracy_val(test_set, 
#                                    actual, 
#                                    self_test, 
#                                    actual_in, 
#                                    metrics = 'sse')

xtest = test_set
ytest = actual
xtrain = self_test
ytrain = actual_in

# sort data once (sort x based on y, then sort y)
xtest = [xtest for _, xtest in sorted(zip(ytest, xtest))]
ytest = sorted(ytest)

xtrain = [xtrain for _, xtrain in sorted(zip(ytrain, xtrain))]
ytrain = sorted(ytrain)


# predictions show the leaf id falling
predictions = CARTmodel.tree_preds(xtest)
predictions_in = CARTmodel.tree_preds(xtrain)

leaf_dict = dict((str(l),[]) for l in CARTmodel.leaves)
leaf_dict_in = dict((str(l),[]) for l in CARTmodel.leaves)

#print(self.leaves)

for l in range(len(CARTmodel.leaves)):
    leaf_dict[str(CARTmodel.leaves[l])] = [ytest[i] for i in range(len(ytest)) if predictions[i] == l]
    leaf_dict_in[str(CARTmodel.leaves[l])] = [ytrain[i] for i in range(len(ytrain)) if predictions_in[i] == l]


total_crps = 0
total_crps_new = 0
total_crps_new2 = 0
for key, val in leaf_dict.items():
    ## key is not sorted, sort first:
    leaf = sorted(ast.literal_eval(key))
    leaf_ytrain = list(Counter(leaf).keys())
    leaf_ytrain_freq = list(Counter(leaf).values())
    
    ## loop in ytrain
    crps_2 = 0.0
    for j, x1 in enumerate(leaf_ytrain):
        s = 0.0
        for i, x2 in enumerate(leaf_ytrain):
            s += abs(x1-x2)/(pow(len(leaf_ytrain),2)*2)*leaf_ytrain_freq[i]
        crps_2 += s*leaf_ytrain_freq[j]
    
    ## val is already sorted:
    leaf_ytest = sorted(list(Counter(val).keys()))
    leaf_ytest_freq = sorted(list(Counter(val).values()))
    
    ## loop in ytest
    crps_1 = 0.0
    for j, y in enumerate(leaf_ytest):
        s = 0.0
        for i, x in enumerate(leaf_ytrain):
            s += abs(y-x)/len(leaf_ytrain)*leaf_ytrain_freq[i]
        crps_1 += s*leaf_ytest_freq[j]        
    total_crps += crps_1 - crps_2*len(val)   
    
    ## just one loop
    for j, y in enumerate(leaf_ytest):
        crps_y = 0.0
        for i, x in enumerate(leaf_ytrain):
            crps_y += (abs(y-x)/len(leaf_ytrain) + x/len(leaf_ytrain) - 2*i*x/(len(leaf_ytrain)*(len(leaf_ytrain)-1)))*leaf_ytrain_freq[i]
        total_crps_new += crps_y*leaf_ytest_freq[j]
    
    ## just one loop
    for j, y in enumerate(leaf_ytest):
        crps_y2 = 0.0
        for i, x in enumerate(leaf_ytrain):
            crps_y2 += 2*(x-y)*(len(leaf_ytrain)*(x>y)-(i+1)+0.5)*leaf_ytrain_freq[i]/(len(leaf_ytrain)*len(leaf_ytrain))
        total_crps_new2 += crps_y2*leaf_ytest_freq[j]
        
        
## in-sample
total_crps = 0
total_crps_new = 0
total_crps_new2 = 0
for key, val in leaf_dict_in.items():
    ## key is not sorted, sort first:
    leaf = sorted(ast.literal_eval(key))
    leaf_ytrain = list(Counter(leaf).keys())
    leaf_ytrain_freq = list(Counter(leaf).values())


    crps_2 = 0.0
    for j, leaf_point_q in enumerate(leaf_ytrain):
        s = 0.0
        for i, leaf_point in enumerate(leaf_ytrain):
            s += abs(leaf_point_q-leaf_point)*leaf_ytrain_freq[i]
        crps_2 += s*leaf_ytrain_freq[j]
    total_crps += crps_2/(2*len(leaf_ytrain))        
    
    for j, y in enumerate(leaf_ytrain):
        crps_y = 0.0
        for i, x in enumerate(leaf_ytrain):
            crps_y += 2*(x-y)*(len(leaf_ytrain)*(x>y)-(i+1)+0.5)*leaf_ytrain_freq[i]/(len(leaf_ytrain)*len(leaf_ytrain))
        total_crps_new += crps_y*leaf_ytrain_freq[j]
    
    ## since the values are repeated half-way, just compute for half of ytrain and multiply by 2
    for j, y in enumerate(leaf_ytrain):
        if j < len(leaf_ytrain)/2:
            crps_y = 0.0
            for i, x in enumerate(leaf_ytrain):
                crps_y += 2*(x-y)*(len(leaf_ytrain)*(x>y)-(i+1)+0.5)*leaf_ytrain_freq[i]/(len(leaf_ytrain)*len(leaf_ytrain))
            total_crps_new2 += crps_y*leaf_ytrain_freq[j]
    total_crps_new2 = total_crps_new2*2
        
        