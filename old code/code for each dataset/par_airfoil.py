#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:09:37 2019

@author: sarashashaani

latest developments from realdata_parallelexperiments_new for dataset: airfoil
"""
import numpy as np
from scipy import random as sr
from random import sample
import ast
import time
from csv import writer
from joblib import Parallel, delayed
from collections import Counter

# Evaluate an algorithm using a cross validation split
# Changing this to have an output as dict, with each key being a tree_method, and each value a list of 5 (tree_evals) lists of size 2 (out and in)
def evaluate_algorithm(dataset):
    global methods, leaves
    evals_dict = {}
    train_set = dataset[0]    
    """ TEST SET """    
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
          
    for tree_method in methods:
        evals_dict[tree_method] = {}
        tree = decision_tree(train_set, tree_method)
        leaves = []
        leaves = leaves_list(tree, 0)
        predicted = tree_preds(tree, test_set)
        predicted_in = tree_preds(tree, self_test)
        actual = [row[-1] for row in dataset[1]]
        actual_in = [row[-1] for row in dataset[0]]
        
        leaf_dict = dict((str(l),[]) for l in leaves)
        leaf_dict_in = dict((str(l),[]) for l in leaves)
        for l in range(len(leaves)):
            leaf_dict[str(leaves[l])] = [actual[i] for i in range(len(actual)) if predicted[i] == l]
            leaf_dict_in[str(leaves[l])] = [actual_in[i] for i in range(len(actual_in)) if predicted_in[i] == l]    
        
        for eval_method in methods:
            eval_new = [accuracy_funcs(eval_method, leaf_dict)]
            eval_new += [accuracy_funcs(eval_method, leaf_dict_in)]
            evals_dict[tree_method][eval_method] = eval_new
    return evals_dict


# List of data points in all leaves 
def leaves_list(node, depth=0):
    global leaves
    if isinstance(node, dict):
        leaves_list(node['left'], depth+1)
        leaves_list(node['right'], depth+1)
    else:
        leaves.append(node)
    return leaves
 
# Classification and Regression Tree Algorithm; Output: predictions as the 
#   whole data in the leaves for each test data point
def decision_tree(train, tree_method):
    global max_depth
    tree = build_tree(train, tree_method)
    return tree

def tree_preds(tree, test_set):
    global leaves
    predictions = list()
    for row in test_set:
        prediction = predict(tree, row)
        predictions.append(leaves.index(prediction))
    return predictions

def accuracy_funcs(method, leaf_dict):
    if method == 'sse':
        return accuracy_sse_usefreq(leaf_dict)
    if method == 'crps':
        return accuracy_crps_usefreq(leaf_dict)
    if method == 'dss':
        return accuracy_dss_usefreq(leaf_dict)
    if method == 'is1':
        return accuracy_is1_usefreq(leaf_dict)


def accuracy_sse_usefreq(leaf_dict):
    total_sse = 0
    for key, val in leaf_dict.items():
        leaf = ast.literal_eval(key)
        avg = np.mean(leaf)
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_sse += pow(point - avg, 2)*rv[j]
            
    return total_sse

def accuracy_crps_usefreq(leaf_dict):  
    global crps_quantiles
    total_crps = 0
    for key, val in leaf_dict.items():
        leaf = sorted(ast.literal_eval(key))
                
        qe, pe = ecdf(np.array(leaf))
        leaf_reduced = [qe[-1]]
        if len(leaf)>2:
            inc_p = 1/min(crps_quantiles-2,(len(leaf)-2))
            inds = [next(x[0] for x in enumerate(pe) if x[1] > (i+.15)*inc_p) for i in range(min(crps_quantiles-2,len(leaf)-2))]
            leaf_reduced += [qe[i] for i in reversed(inds)]
        if len(leaf)>1:
            leaf_reduced += [qe[0]]
            
        x = list(Counter(leaf_reduced).keys()) # equals to list(set(targets))
        r = list(Counter(leaf_reduced).values())
        m = len(leaf_reduced)
#        
        total_crps_g = 0.0
        for j, point in enumerate(x):
            s = 0.0
            for i, leaf_point in enumerate(x):
                s += abs(point-leaf_point)*r[i]
            total_crps_g += s*r[j] 
        total_crps_g /=  (2*pow(m,2))

        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())
        mv = len(val)
        
        total_crps = 0.0
        for j, point in enumerate(xv):
            s = 0.0
            for i, leaf_point in enumerate(x):
                s += abs(point-leaf_point)*r[i]
            total_crps += s*rv[j]
        total_crps = (total_crps-total_crps_g)/mv
    return total_crps


def accuracy_dss(leaf_dict):  
    total_ms = 0
    for key, val in leaf_dict.items():
        leaf = ast.literal_eval(key)
        mhat = np.mean(leaf)
        vhat = np.var(leaf)
#        shat = max(np.std(leaf),.00001)
        for point in val:
            total_ms += pow(point - mhat,2)/vhat+np.log(vhat)      
    return total_ms

def accuracy_dss_usefreq(leaf_dict):  
    total_ms = 0
    for key, val in leaf_dict.items():
        leaf = ast.literal_eval(key)
        mhat = np.mean(leaf)
        vhat = np.var(leaf)
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_ms += (pow(point - mhat,2)/vhat+np.log(vhat))*rv[j]

    return total_ms

def accuracy_is1(leaf_dict):  
    global alpha
    total_is = 0
    for key, val in leaf_dict.items():
        leaf = sorted(ast.literal_eval(key))
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]
        for point in val:
            total_is += u+(point-u)*(point>=u)/alpha
    return total_is

def accuracy_is1_usefreq(leaf_dict):  
    global alpha
    total_is = 0
    for key, val in leaf_dict.items():
        leaf = sorted(ast.literal_eval(key))
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_is += (u+(point-u)*(point>=u))*rv[j]

    return total_is


# Split a dataset based on an attribute and an attribute value
# This is candidate split, so all we do here is to devide the dataset into 
#   left (attribute <= attribute value) and right (o.w.)
#   left (attribute == attribute value) and right (o.w.)  if equal (for categorical vars)   
def test_split(index, value, train_set, equal):
    left, right = list(), list()
    if equal:
        for row in train_set:
            if row[index] == value:
                left.append(row)
            else:
                    right.append(row)
    else:
        for row in train_set:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
    return left, right

def new_split_funcs(method, groups, notparent):
    if method == 'sse':
        return sse_for_new_split_usefreq(groups,notparent)
    if method == 'crps':
        return crps_for_new_split_usefreq(groups,notparent)        
    if method == 'dss':
        return dss_for_new_split_usefreq(groups,notparent)
    if method == 'is1':
        return is1_for_new_split_usefreq(groups,notparent)

def sse_for_new_split_usefreq(groups,notparent):
    sse = 0.0
    if notparent:
        for group in groups:
            targets = sorted(np.asarray([row[-1] for row in group]))
            x = list(Counter(targets).keys()) # equals to list(set(targets))
            r = list(Counter(targets).values())
            
            mean_target = sum(targets)/float(len(group))
            
            for j, point in enumerate(x):
                sse += pow(point-mean_target,2)*r[j]
    else:
        targets = sorted(np.asarray([row[-1] for row in groups]))
        x = list(Counter(targets).keys()) # equals to list(set(targets))
        r = list(Counter(targets).values())
        
        mean_target = sum(targets)/float(len(groups))
        
        for j, point in enumerate(x):
            sse += pow(point-mean_target,2)*r[j]

    return sse


# Find the empirical cdf of a sample, Outcome: quantiles and cumulative probabilities
def ecdf(sample):
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

def crps_for_new_split_usefreq(groups,notparent):
    global crps_quantiles
    total_crps = 0
    if notparent:
        total_m = 0
        for group in groups:
            targets = sorted(np.asarray([row[-1] for row in group]))
            
            qe, pe = ecdf(np.array(targets))
            targets_reduced = [qe[-1]]
            if len(targets)>2:
                inc_p = 1/min(crps_quantiles-2,(len(targets)-2))
                inds = [next(x[0] for x in enumerate(pe) if x[1] > (i+.15)*inc_p) for i in range(min(crps_quantiles-2,len(targets)-2))]
                targets_reduced += [qe[i] for i in reversed(inds)]
            if len(targets)>1:
                targets_reduced += [qe[0]]
                
            x = list(Counter(targets_reduced).keys()) # equals to list(set(targets))
            r = list(Counter(targets_reduced).values())
            m = len(targets_reduced)
#            
            total_m += m
            total_crps_g = 0.0
            for j, point in enumerate(x):
                s = 0.0
                for i, leaf_point in enumerate(x):
                    s += abs(point-leaf_point)*r[i]
                total_crps_g += s*r[j] 
            total_crps_g /=  (2*pow(m,2)) 
                        
            total_crps += total_crps_g*m
        total_crps /= total_m
                
    else:
        targets = sorted(np.asarray([row[-1] for row in groups]))

        qe, pe = ecdf(np.array(targets))
        targets_reduced = [qe[-1]]
        crps_quantiles = 1000
        if len(targets)>2:
            inc_p = 1/min(crps_quantiles-2,(len(targets)-2))
            inds = [next(x[0] for x in enumerate(pe) if x[1] > (i+.15)*inc_p) for i in range(min(crps_quantiles-2,len(targets)-2))]
            targets_reduced += [qe[i] for i in reversed(inds)]
        if len(targets)>1:
            targets_reduced += [qe[0]]
            
        x = list(Counter(targets_reduced).keys()) # equals to list(set(targets))
        r = list(Counter(targets_reduced).values())
        m = len(targets_reduced)

        total_crps = 0.0
        for j, point in enumerate(x):
            s = 0.0
            for i, leaf_point in enumerate(x):
                s += abs(point-leaf_point)*r[i]
            total_crps += s*r[j]     
        total_crps /= (2*pow(m,2))        
    return total_crps  

def dss_for_new_split_usefreq(groups,notparent):
    global alpha
    dss = 0.0
    if notparent:
        for group in groups:
            targets = np.asarray([row[-1] for row in group])
            mhat = np.mean(targets)
            vhat = np.var(targets)
            
            x = list(Counter(targets).keys()) # equals to list(set(targets))
            r = list(Counter(targets).values())

            for j, point in enumerate(x):
                dss += (pow(point - mhat,2)/vhat+np.log(vhat))*r[j]
    else:
        targets = np.asarray([row[-1] for row in groups])
        mhat = np.mean(targets)
        vhat = np.var(targets)
        
        x = list(Counter(targets).keys()) # equals to list(set(targets))
        r = list(Counter(targets).values())

        for j, point in enumerate(x):
            dss += (pow(point - mhat,2)/vhat+np.log(vhat))*r[j]
            
    return dss

def is1_for_new_split_usefreq(groups,notparent):
    global alpha
    is1 = 0.0
    if notparent:
        for group in groups:
            targets = sorted(np.asarray([row[-1] for row in group]))
            u = targets[int(np.ceil((1-alpha)*len(targets)))-1]
            
            x = list(Counter(targets).keys()) # equals to list(set(targets))
            r = list(Counter(targets).values())
    
            for j, point in enumerate(x):
                is1 += (u+(point-u)*(point>=u)/alpha)*r[j]
    else:
        targets = sorted(np.asarray([row[-1] for row in groups]))
        u = targets[int(np.ceil((1-alpha)*len(targets)))-1]

        x = list(Counter(targets).keys()) # equals to list(set(targets))
        r = list(Counter(targets).values())

        for j, point in enumerate(x):
            is1 += (u+(point-u)*(point>=u)/alpha)*r[j]
    return is1


# Select the best split point for a dataset
#   based on tree_method: crps or sse; start by b_score before split and 
#   search for lowest score across all candidate splits
def get_split(train_set, tree_method):
    global min_node_size, num_quantiles, x_dim, tol, is_cat, cov_uniqvals
    b_index, b_value, b_groups = 999, 999, None
    b_score = new_split_funcs(tree_method, train_set, 0)
    first_val = 0  
    split_occurs = 0

    for index in range(x_dim):
        qe, pe = ecdf(column(train_set,index))
        if is_cat[index]:# and len(unique_vals) <= 25:
            tocheck_val = qe
            equal = 1
        elif len(qe) < num_quantiles:
            tocheck_val = qe
            equal = 0
        else:
            inc_p = 1/(num_quantiles+1)
            inds = [next(x[0] for x in enumerate(pe) if x[1] > i*inc_p) for i in range(1,(num_quantiles+1))] 
            tocheck_val = list(sorted(set([qe[i] for i in inds])))
            equal = 0        
        for val in tocheck_val:
            groups = test_split(index, val, train_set, equal)
            if len(groups[0]) >= min_node_size and len(groups[1]) >= min_node_size:
                measure =  new_split_funcs(tree_method, groups, 1)
                if not first_val:
                    first_val = 1
                    if b_score < measure:
                        print("monotonicity violated - "+str(tree_method)+" - variable "+str(index))
#                        log_file.write("monotonicity violated - "+str(tree_method)+" - variable "+str(val))
                    b_score = max(b_score,measure)
                if split_occurs:
                    check_tol = 0
                else:
                    check_tol = tol

                if measure <= b_score*(1-check_tol):                    
                    split_occurs = 1
                    b_index, b_value, b_score, b_groups = index, val, measure, groups
    if not split_occurs:
        print("no improvement - "+str(tree_method))
        log_file.write("no improvement - "+str(tree_method))
    return {'index':b_index, 'value':b_value, 'groups':b_groups}   

# Return the observaions in the leaf    
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return outcomes
 
    
# Create child splits for a node 
#   or make terminal (leaf) if (1) no split improves the current node, 
#   or (2) the depth of the tree is maxed or (3) the volume of the node before split 
#   is less than twice of the min_node_size (minimum data points in any node)
def split(node, depth, tree_method):
    global min_node_size, max_depth
    if node['groups']:
        left, right = node['groups']
        del(node['groups'])
    else:
        print('NOTHING')
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    # process left child
    if len(left) < 3*min_node_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, tree_method)
        split(node['left'], depth+1, tree_method)
    # process right child
    if len(right) < 3*min_node_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, tree_method)
        split(node['right'], depth+1, tree_method)



# Build a decision tree
# Start with the root to get the first split, then call the recursice Split function
def build_tree(train_set, tree_method):
    global max_depth
    root = get_split(train_set, tree_method)
    split(root, 1, tree_method)
    print_tree(root, depth=0)
    return root
    
# Print a decision tree
def print_tree(node, depth=0):
    global is_cat
    if isinstance(node, dict):
        if is_cat[node['index']]:
            print('%s[X%d = %d]' % ((depth*' ', (node['index']+1), int(node['value']))))
#            log_file.write('%s[X%d = %d]' % ((depth*' ', (node['index']+1), int(node['value']))))
        else:
            print('%s[X%d < %.4f]' % ((depth*' ', (node['index']+1), node['value'])))
#            log_file.write('%s[X%d < %.4f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', len(node)))) 
#        log_file.write('%s[%s]' % ((depth*' ', len(node)))) 
    
# Make a prediction with a decision tree
# Return the node (the entire leaf, not just a summary)
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


def column(matrix, i):
    return [row[i] for row in matrix]

""" evaluate algorithm """
def OneRep(k):
    sr.seed(k+2010)

    holdout_size = int(len(rows)/2) 
    test_index = list(sample(range(len(rows)),holdout_size))
    train_index = list(set(range(len(rows))) - set(test_index))
    
    train_set = [rows[index] for index in train_index]    
    test_set = [rows[index] for index in test_index]
    dataset = [train_set,test_set]
    
    total_time = time.time()
    scores = evaluate_algorithm(dataset)
    total_time = time.time() - total_time
    print("Rep "+str(k)+" completed in "+str(round(total_time,2))+" sec.")
    log_file.write("\nRep "+str(k)+" completed in "+str(round(total_time,2))+" sec.")

    return scores

global min_node_size, max_depth, methods, num_quantiles, crps_quantiles, alpha, tol, x_dim, is_cat, cov_uniqvals, leaves
leaves = []
tol = 0
crps_quantiles = 1000

## inputs
max_depth = 1
min_node_size = 100
num_quantiles = 20
total_reps = 10
alpha = .2

methods = ["crps", "dss", "is1", "sse"]
params = str(max_depth)+str(min_node_size)+str(num_quantiles)+str(total_reps)+str(alpha)
#,"casp","ccpp","co2","divy","fb","matt","methane","pipeP","pipeT","yield"]

total_time = time.time()
data_title = "airfoil"
#data_title = "matt"
directory = "/home/sshasha2/"
#directory = "/Users/sarashashaani/Documents/"
datafile = directory+"data/test_"+data_title+".txt"
log_file = open(directory+"results/log_"+data_title+"_"+params+".txt", 'a+')


with open (datafile, 'r') as f: # use with to open your files, it close them automatically
    rows = [x.split() for x in f]    
rows = rows[1:]
for i in range(len(rows)):
    rows[i] = [float(x) for x in rows[i]]
x_dim = len(rows[0])-1

is_cat = []
cov_uniqvals = []
for i in range(x_dim):
    unique_vals = list(sorted(set(column(rows,i))))
    cov_uniqvals += [unique_vals]
    if len(unique_vals) <= 2:#len(rows)/len(unique_vals) > 100:
        is_cat += [1]
    else:
        is_cat += [0]

results = Parallel(n_jobs=min(total_reps,20))(delayed(OneRep)(rep_no) for rep_no in range(total_reps))
#results = Parallel(n_jobs=2)(delayed(OneRep)(rep_no) for rep_no in range(total_reps))

csv_dict_out = {method: [] for method in methods}
csv_dict_in = {method: [] for method in methods}

for e_method in methods:
    reps_list = []
    reps_list_in = []
    for rep in range(total_reps):
        rep_list = []
        rep_list_in = []
        for t_method in methods:
            rep_list += [round(results[rep][t_method][e_method][0],2)]
            rep_list_in += [round(results[rep][t_method][e_method][1],2)]
        reps_list += [rep_list]
        reps_list_in += [rep_list_in]
    csv_dict_out[e_method] += reps_list 
    csv_dict_in[e_method] += reps_list_in             

    with open(directory+"results/"+data_title+"_"+e_method+"_out_"+params+".csv", "w") as f:
        w = writer(f)
        w.writerows(csv_dict_out[e_method])
    with open(directory+"results/"+data_title+"_"+e_method+"_in_"+params+".csv", "w") as f:
        w = writer(f)
        w.writerows(csv_dict_in[e_method])
        
total_time = time.time() - total_time
print(data_title+" completed in "+str(round(total_time,2))+" sec.")
log_file.write("\n"+data_title+" completed in "+str(round(total_time,2))+" sec.")

for e_method in methods:
    print("EVALUATION of "+data_title+" with "+e_method)
    log_file.write("\nEVALUATION of "+data_title+" with "+e_method)
    for i, t_method in enumerate(methods):
            print("tree built via "+t_method+", in-sample mean: "+str(round(np.mean(column(csv_dict_in[e_method], i)),2)))
            log_file.write("\ntree built via "+t_method+", in-sample mean: "+str(round(np.mean(column(csv_dict_in[e_method], i)),2)))
    for i, t_method in enumerate(methods):
            print("tree built via "+t_method+", out-of-sample mean: "+str(round(np.mean(column(csv_dict_out[e_method], i)),2)))
            log_file.write("\ntree built via "+t_method+", out-of-sample mean: "+str(round(np.mean(column(csv_dict_out[e_method], i)),2)))