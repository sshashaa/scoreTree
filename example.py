#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:57:43 2022

@author: ozgesurer
"""
from scoreCART import scoreCART
from random import sample

def column(matrix, i):
    return [row[i] for row in matrix]

# inputs
max_depth = 3
min_node_size = 100
num_quantiles = 20
total_reps = 4 # just using 4 for now because running in parallel on local computer
alpha = .2
tol = 0
data_title = "airfoil"

#methods = ["crps", "dss", "is1", "sse"]
methods = ["sse"]
params = str(max_depth)+str(min_node_size)+str(num_quantiles)+str(total_reps)+str(alpha)

datafile = "data/test_" + data_title + ".txt"
log_file = open("log_" + data_title + "_" + params+".txt", 'a+')
    
with open (datafile, 'r') as f: # use with to open your files, it close them automatically
    rows = [x.split() for x in f]    
rows = rows[1:]
for i in range(len(rows)):
    rows[i] = [float(x) for x in rows[i]]
x_dim = len(rows[0])-1

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
        
# Fit the tree model
CARTmodel = scoreCART(methods[0], 
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
