#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:21:28 2019

@author: sarashashaani
airfoil dataset 
most recent last_working_withuf 
max level = 3
min node size = 100
split quantile = 20
crps quantil = N/A

!!!! before usign on server, change directory, and matplotlib.use('Agg')
"""

import numpy as np
from scipy import random as sr
from random import sample
import ast
import time
from csv import writer
from joblib import Parallel, delayed
from collections import Counter
#import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt



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
#            print(eval_method+' eval: '+str(eval_new))
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
        return accuracy_sse(leaf_dict)
    if method == 'crps':
        return accuracy_crps(leaf_dict)
    if method == 'dss':
        return accuracy_dss(leaf_dict)
    if method == 'is1':
        return accuracy_is1(leaf_dict)

# Evaluation metric: SSE; Input is the actual data and the all the observations 
#   of the leaf each data point falls in (predicted)
def accuracy_sse(leaf_dict):
    total_sse = 0
    for key, val in leaf_dict.items():
        leaf = ast.literal_eval(key)
        avg = np.mean(leaf)
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_sse += pow(point - avg, 2)*rv[j]
    return total_sse

def accuracy_crps(leaf_dict):  
    total_crps = 0    ## crps old with freq  -- this is correct
    for key, val in leaf_dict.items(): # key is X and val is y
        leaf = ast.literal_eval(key)
        
        x = list(Counter(leaf).keys())
        r = list(Counter(leaf).values())
        
        crps_2 = 0.0
        for j, leaf_point_q in enumerate(x):
            s = 0.0
            for i, leaf_point in enumerate(x):
                s += abs(leaf_point_q-leaf_point)/(pow(len(leaf),2)*2)*r[i]
            crps_2 += s*r[j]
        
        xv = list(Counter(val).keys())
        rv = list(Counter(val).values())
        
        crps_1 = 0.0
        for j, leaf_point_q in enumerate(xv):
            s = 0.0
            for i, leaf_point in enumerate(x):
                s += abs(leaf_point_q-leaf_point)*r[i]
            crps_1 += s*rv[j]        
        total_crps += crps_1/len(leaf) - crps_2*len(val)
    return total_crps


def accuracy_dss(leaf_dict):  
    total_dss = 0
    for key, val in leaf_dict.items():
        leaf = ast.literal_eval(key)
        mhat = np.mean(leaf)
        vhat = max(np.var(leaf),.1)
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_dss += (pow(point - mhat,2)/vhat+np.log(vhat))*rv[j]
    return total_dss

def accuracy_is1(leaf_dict):
    global alpha
    total_is = 0
    for key, val in leaf_dict.items():
        leaf = sorted(ast.literal_eval(key))
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_is += (u+(point-u)*(point>=u)/alpha)*rv[j]
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
        return sse_for_new_split(groups,notparent)
    if method == 'crps':
        return crps_for_new_split(groups,notparent)
    if method == 'dss':
        return dss_for_new_split(groups,notparent)
    if method == 'is1':
        return is1_for_new_split(groups,notparent)

#  SSE of a a new splitted point; if nonparent it is two nodes (left, right)
#   if parent, only one node
def sse_for_new_split(groups,notparent):
    sse = 0.0
    if notparent:
        for group in groups:
            mean_target = sum([row[-1] for row in group])/float(len(group))
            sse += sum([pow(row[-1]-mean_target,2) for row in group])
    else:
        mean_target = sum([row[-1] for row in groups])/float(len(groups))
        sse = sum([pow(row[-1]-mean_target,2) for row in groups])  
    return sse

# Find the empirical cdf of a sample, Outcome: quantiles and cumulative probabilities
def ecdf(sample):
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

def crps_for_new_split(groups,notparent):
    total_crps = 0          
    if notparent:
        for group in groups:
            targets = np.asarray([row[-1] for row in group])            
            x = list(Counter(targets).keys())
            r = list(Counter(targets).values())
            
            crps_2 = 0.0
            for j, leaf_point_q in enumerate(x):
                s = 0.0
                for i, leaf_point in enumerate(x):
                    s += abs(leaf_point_q-leaf_point)*r[i]
                crps_2 += s*r[j]            
            total_crps += crps_2/(2*len(targets))
    else:
        targets = np.asarray([row[-1] for row in groups])
        x = list(Counter(targets).keys())
        r = list(Counter(targets).values())
        
        crps_2 = 0.0
        for j, leaf_point_q in enumerate(x):
            s = 0.0
            for i, leaf_point in enumerate(x):
                s += abs(leaf_point_q-leaf_point)*r[i]
            crps_2 += s*r[j]
        total_crps += crps_2/(2*len(targets))
    return total_crps  

def dss_for_new_split(groups,notparent):
    dss = 0.0
    if notparent:
        for group in groups:
            targets = np.asarray([row[-1] for row in group])
            mhat = np.mean(targets)
            vhat = max(np.var(targets),.1)
            dss += (np.log(vhat)*len(targets)+ sum([pow(x-mhat,2) for x in targets])/vhat)
    else:
        targets = np.asarray([row[-1] for row in groups])
        mhat = np.mean(targets)
        vhat = max(np.var(targets),.1)
        dss += (np.log(vhat)*len(targets)+ sum([pow(x-mhat,2) for x in targets])/vhat)
    return dss

def is1_for_new_split(groups,notparent):
    global alpha
    is1 = 0.0
    if notparent:
        for group in groups:
            targets = sorted(np.asarray([row[-1] for row in group]))
            u = targets[int(np.ceil((1-alpha)*len(targets)))-1]
            is1 += (u*len(targets)+sum([(x-u)*(x>=u) for x in targets])/alpha)
    else:
        targets = sorted(np.asarray([row[-1] for row in groups]))
        u = targets[int(np.ceil((1-alpha)*len(targets)))-1]
        is1 += (u*len(targets)+sum([(x-u)*(x>=u) for x in targets])/alpha)
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
                        log_file.write("monotonicity violated - "+str(tree_method)+" - variable "+str(val))
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
    print("tree_method "+tree_method+"\n###########################")
    print_tree(root, depth=0)
    return root
    
# Print a decision tree
def print_tree(node, depth=0):
    global is_cat
    if isinstance(node, dict):
        if is_cat[node['index']]:
            print('%s[X%d = %d]' % ((depth*' ', (node['index']+1), int(node['value']))))
        else:
            print('%s[X%d < %.4f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', len(node)))) 
    
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
    train_index = list(sample(range(len(rows)),holdout_size))
    test_index = list(set(range(len(rows))) - set(train_index))
    
    train_set = [rows[index] for index in train_index]    
    test_set = [rows[index] for index in test_index]
    dataset = [train_set,test_set]
    
    total_time = time.time()
    scores = evaluate_algorithm(dataset)
    total_time = time.time() - total_time
    print("Rep "+str(k)+" completed in "+str(round(total_time,2))+" sec.")
    log_file.write("\nRep "+str(k)+" completed in "+str(round(total_time,2))+" sec.")
    
    return scores

def set_box_colors(bp):
    colors = ['red', 'purple', 'blue', 'green']
    elements_1 = ['boxes','fliers']
    elements_2= ['caps','whiskers']
    for elem in elements_1:
        for idx in range(len(bp[elem])):
            plt.setp(bp[elem][idx], color=colors[idx])
    for elem in elements_2:
        for idx in range(int(len(bp[elem])/2)):
            plt.setp(bp[elem][2*idx], color=colors[idx])
            plt.setp(bp[elem][2*idx+1], color=colors[idx])


def plot(e_method):
    global params
    dataset1 = np.transpose(np.array(csv_dict_in[e_method]))
    dataset2 = np.transpose(np.array(csv_dict_out[e_method]))
#
    fig = plt.figure()
    fig.suptitle(data_title)
    
    print(e_method+"-evaluated")
    for j, t_method in enumerate(methods):
        print(t_method+"-built tree, in-sample mean: "+str(round(np.mean(dataset1[j,:]),2)))
        log_file.write("\n"+t_method+"-built tree, in-sample mean: "+str(round(np.mean(dataset1[j,:]),2)))
    plt.subplot(1, 2, 1)
    plt.title('in-sample '+str(e_method), fontsize=20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    bp = plt.boxplot(dataset1.tolist(),positions = [1, 2, 3, 4], widths = 0.9)
    set_box_colors(bp)
    frame1 = plt.gca()  
    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)
    for j, t_method in enumerate(methods):
        print(t_method+"-built tree, out-of-sample mean: "+str(round(np.mean(dataset2[j,:]),2)))
        log_file.write("\n"+t_method+"-built tree, out-sample mean: "+str(round(np.mean(dataset2[j,:]),2)))
    plt.subplot(1, 2, 2)
    plt.title('out-of-sample '+str(e_method), fontsize=20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    bp = plt.boxplot(dataset2.tolist(),positions = [1, 2, 3, 4], widths = 0.9)
    set_box_colors(bp)
    frame1 = plt.gca()  
    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)

    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.savefig(directory+"results/"+data_title+"_"+e_method+"_"+params+".png".format(1))

    
global min_node_size, max_depth, methods, num_quantiles, alpha, tol, x_dim, is_cat, cov_uniqvals, leaves, params
leaves = []
tol = 0

# inputs
max_depth = 2
min_node_size = 30
num_quantiles = 5
total_reps = 10
alpha = .2
data_title = "airfoil"

methods = ["crps", "dss", "is1", "sse"]
params = str(max_depth)+str(min_node_size)+str(num_quantiles)+str(total_reps)+str(alpha)
total_time = time.time()

import os
os.getcwd()
directory = '/Users/sarashashaani/Google Drive/ScoringRules/'

#directory = "/home/sshasha2/"


datafile = directory+"data/test_"+data_title+".txt"
log_file = open(directory+"log_"+data_title+"_"+params+".txt", 'a+')
        
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

total_time = time.time() 
results = Parallel(n_jobs=min(total_reps,20))(delayed(OneRep)(rep_no) for rep_no in range(total_reps))
#print(results)
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
    plot(e_method)

