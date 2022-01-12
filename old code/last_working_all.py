#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:12:37 2019

@author: sarashashaani

Modified on Mon May 27 15:50:10 2019

1. using new random seed use 
2. making parameters function of the dataset
3. changing the name of IS1 to IS-h
4. adding IS-l by negating the dataset and finding the same quantile 
    and then multiplying the interval by -1
5. saving the paired difference between sse-based tree and score-based tree
    and its squared value
6. calculating the normalied paired differences (mean divided by std error)
7. saving the score (crps, for now) after each split


Modified on Fri Jun 21 21:09:36 2019
creating plots of normalized residuals for each score (datasets on x-axis)

"""

import numpy as np
import ast
#import sys
import time
from csv import writer
from joblib import Parallel, delayed
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csv import reader

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
        total_time = time.time() 
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
        print(time.time() - total_time)
        total_time = time.time() 
        for eval_method in methods:
            eval_new = [accuracy_funcs(eval_method, leaf_dict)]
            eval_new += [accuracy_funcs(eval_method, leaf_dict_in)]
            evals_dict[tree_method][eval_method] = eval_new
#            print(eval_method+' eval: '+str(eval_new))
        print(time.time() - total_time)
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
    if method == 'ish':
        return accuracy_ish(leaf_dict)
    if method == 'isl':
        return accuracy_isl(leaf_dict)

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

def accuracy_ish(leaf_dict):
    global alpha
    total_is = 0
    for key, val in leaf_dict.items():
        leaf = sorted(ast.literal_eval(key))
        # leaf = sorted(column(rows,x_dim))
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_is += (u+(point-u)*(point>=u)/alpha)*rv[j]
    return total_is

def accuracy_isl(leaf_dict):
    global alpha
    total_is = 0
    for key, val in leaf_dict.items():
        leaf = sorted([c * -1 for c in ast.literal_eval(key)])
        # leaf = sorted([c * -1 for c in column(rows,x_dim)])
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]*-1
        
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
    if method == 'ish':
        return ish_for_new_split(groups,notparent)
    if method == 'isl':
        return isl_for_new_split(groups,notparent)

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

def ish_for_new_split(groups,notparent):
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

def isl_for_new_split(groups,notparent):
    global alpha
    is1 = 0.0
    if notparent:
        for group in groups:
            targets = sorted(np.asarray([row[-1] for row in group]))
#            u = targets[int(np.ceil((1-alpha)*len(targets)))-1]
            leaf = sorted([c * -1 for c in column(group,x_dim)])
            u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]*-1
            is1 += (u*len(targets)+sum([(x-u)*(x>=u) for x in targets])/alpha)
    else:
        leaf = sorted([c * -1 for c in column(groups,x_dim)])
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]*-1
        targets = sorted(np.asarray([row[-1] for row in groups]))
#        u = targets[int(np.ceil((1-alpha)*len(targets)))-1]
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
        
""" evaluate algorithm """
def OneRep(k):
    holdout_size = int(len(rows)/2) 
    rng = np.random.RandomState(k)
    train_index = set(rng.choice(len(rows),size=holdout_size,replace=False))
    test_index = set(range(len(rows))) - train_index
    train_set = [rows[index] for index in train_index]    
    test_set = [rows[index] for index in test_index]
    dataset = [train_set,test_set]
    
    total_time = time.time()
    scores = evaluate_algorithm(dataset)
    total_time = time.time() - total_time
#    print("Rep "+str(k)+" completed in "+str(round(total_time,2))+" sec.")
    log_file.write("\nRep "+str(k)+" completed in "+str(round(total_time,2))+" sec.")
    
    return scores

  
global min_node_size, max_depth, methods, num_quantiles, alpha, tol, x_dim, is_cat, cov_uniqvals, leaves, params
leaves = []
# params
tol = 0
num_quantiles = 5
total_reps = 20
alpha = .2
data_title = "methane"

def column(matrix, i):
    return [row[i] for row in matrix]


#methods = ["crps", "dss", "sse"]
methods = ["crps", "dss", "ish", "isl", "sse"]
data_titles = ["airfoil","casp","ccpp","co2","divy","fb","methane","wind","yield"]
#data_titles = ["airfoil","ccpp"]
directory = "/home/sshasha2/Documents/"
log_file = open(directory+"log_all.txt", 'a+')
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
plt.style.use('seaborn-whitegrid')   

def set_box_colors(bp):
    elements_1 = ['boxes','fliers']
    elements_2= ['caps','whiskers']
    for elem in elements_1:
        for idx in range(len(bp[elem])):
            plt.setp(bp[elem][idx], color=colors[idx])
    for elem in elements_2:
        for idx in range(int(len(bp[elem])/2)):
            plt.setp(bp[elem][2*idx], color=colors[idx])
            plt.setp(bp[elem][2*idx+1], color=colors[idx])

colors = ['red', 'purple', 'blue', 'orange', 'green']


#in_rep1 = {}
#in_rep2 = {}
#out_rep1 = {}
#out_rep2 = {}
#
#in_rep1_sse = {}
#out_rep1_sse = {}

for data_title in data_titles:
    datafile = directory+"data/test_"+data_title+".txt"        
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
    # other params
    max_depth = min(5, int(np.floor(np.log2(len(rows)/2)*(x_dim/(x_dim+2)))))
    min_node_size = max(50, int(np.ceil(len(set(column(rows,x_dim)))*.01)))
       
    params = str(max_depth)+str(min_node_size)+str(num_quantiles)+str(total_reps)+str(int(alpha*100))
    ## log the results   
    log_file.write("\n\n**************************************************")
    log_file.write("\nDataset: "+data_title)
    log_file.write("\nParameters: "+params)
    
    results = Parallel(n_jobs=min(total_reps,20))(delayed(OneRep)(rep_no) for rep_no in range(total_reps))
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
        
        ## log the results
        log_file.write("\n"+e_method+"-evaluated")
        dataset1 = np.transpose(np.array(csv_dict_in[e_method]))
        dataset2 = np.transpose(np.array(csv_dict_out[e_method]))
        for j, t_method in enumerate(methods):
            log_file.write("\n"+t_method+"-built tree, in-sample mean: "+str(round(np.mean(dataset1[j,:]),2)))
        for j, t_method in enumerate(methods):
            log_file.write("\n"+t_method+"-built tree, out-sample mean: "+str(round(np.mean(dataset2[j,:]),2)))
    
    with open(directory+"results/"+data_title+"_"+e_method+"_out_"+params+".csv", "w") as f:
        w = writer(f)
        w.writerows(csv_dict_out[e_method])
    with open(directory+"results/"+data_title+"_"+e_method+"_in_"+params+".csv", "w") as f:
        w = writer(f)
        w.writerows(csv_dict_in[e_method])
    
    
    ## comparison performance 
#    in_rep1[data_title] = []
#    in_rep2[data_title] = []
#    out_rep1[data_title] = []
#    out_rep2[data_title] = []
#    
#    in_rep1_sse[data_title] = []
#    out_rep1_sse[data_title] = []
#    for j, e_method in enumerate(methods[:-1]):
#        ## difference in e_method
#        ## for paired difference plots
#        out_z = np.array(csv_dict_out[e_method])[:,len(methods)-1]-np.array(csv_dict_out[e_method])[:,j]
#        out_z2 = pow(np.array(csv_dict_out[e_method])[:,len(methods)-1]-np.array(csv_dict_out[e_method])[:,j], 2)       
#        out_me = np.mean(out_z)
#        out_se = np.sqrt(np.sum(out_z2)/(total_reps-1)-(total_reps/(total_reps-1))*pow(out_me,2))/np.sqrt(total_reps)
#        out_rep1[data_title] += [out_me / out_se]
#        out_rep2[data_title] += [sum(out_z>0)/total_reps]
#        
#    
#        in_z = np.array(csv_dict_in[e_method])[:,len(methods)-1]-np.array(csv_dict_in[e_method])[:,j]
#        in_z2 = pow(np.array(csv_dict_in[e_method])[:,len(methods)-1]-np.array(csv_dict_in[e_method])[:,j], 2)       
#        in_me = np.mean(in_z)
#        in_se = np.sqrt(np.sum(in_z2)/(total_reps-1)-(total_reps/(total_reps-1))*pow(in_me,2))/np.sqrt(total_reps)
#        in_rep1[data_title] += [in_me / in_se]
#        in_rep2[data_title] += [sum(in_z>0)/total_reps]
#        
#        
#         ## difference in sse
#        ## for paired difference plots
#        out_z = np.array(csv_dict_out['sse'])[:,len(methods)-1]-np.array(csv_dict_out['sse'])[:,j]
#        out_z2 = pow(np.array(csv_dict_out['sse'])[:,len(methods)-1]-np.array(csv_dict_out['sse'])[:,j], 2)       
#        out_me = np.mean(out_z)
#        out_se = np.sqrt(np.sum(out_z2)/(total_reps-1)-(total_reps/(total_reps-1))*pow(out_me,2))/np.sqrt(total_reps)
#        out_rep1_sse[data_title] += [out_me / out_se]
#    
#        in_z = np.array(csv_dict_in['sse'])[:,len(methods)-1]-np.array(csv_dict_in['sse'])[:,j]
#        in_z2 = pow(np.array(csv_dict_in['sse'])[:,len(methods)-1]-np.array(csv_dict_in['sse'])[:,j], 2)       
#        in_me = np.mean(in_z)
#        in_se = np.sqrt(np.sum(in_z2)/(total_reps-1)-(total_reps/(total_reps-1))*pow(in_me,2))/np.sqrt(total_reps)
#        in_rep1_sse[data_title] += [in_me / in_se]
    
#    ## can run this separately also
#    plt.clf()
#    fig = plt.figure()
##    data_title = 'yield'
#    fig.suptitle(data_title, fontsize=15)
#    plt.subplot(1, 2, 1)
#    plt.title('Normalized Paired Difference', fontsize=10)
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    col = 2 
#    plt.plot(methods[:-1], in_rep1[data_title], '-'+colors[col], label = 'in-sample ')
#    plt.plot(methods[:-1], out_rep1[data_title], '--'+colors[col], label = 'out-sample')
#    plt.legend()
#    frame1 = plt.gca()  
#    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)
#    plt.subplot(1, 2, 2)
#    plt.title('% of Improved Scores', fontsize=10)
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    col = 3
#    plt.plot(methods[:-1], in_rep2[data_title], '-'+colors[col], label = 'in-sample ')
#    plt.plot(methods[:-1], out_rep2[data_title], '--'+colors[col], label = 'out-sample ')
#    plt.legend()
#    frame1 = plt.gca()
#    plt.title('Normalized Paired Difference (by SSE)', fontsize=10)
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    col = 4
#    plt.plot(methods[:-1], in_rep1_sse[data_title], '-'+colors[col], label = 'in-sample ')
#    plt.plot(methods[:-1], out_rep1_sse[data_title], '--'+colors[col], label = 'out-sample ')
#    plt.legend()
#    frame1 = plt.gca()  
#    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)
#    fig.subplots_adjust(hspace=0)
#    fig.tight_layout()
#    plt.savefig(directory+"results/"+data_title+".png".format(1))
        
## plot with datasets in x-axis. 
##           boxplot of normalized residual pairs (score(Tr(score)-score(Tr(SSE))/SD of OS in y-axis, 
## one plot for each score

for j, e_method in enumerate(methods[:-1]):
    out_rep = {}
    fig = plt.figure()
#    fig.suptitle(e_method)
    
    for data_title in data_titles:
        datafile = directory+"data/test_"+data_title+".txt"        
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
        # other params
        max_depth = min(5, int(np.floor(np.log2(len(rows)/2)*(x_dim/(x_dim+2)))))
        min_node_size = max(50, int(np.ceil(len(set(column(rows,x_dim)))*.01)))
           
        params = str(max_depth)+str(min_node_size)+str(num_quantiles)+str(total_reps)+str(int(alpha*100))
        res_filename = directory+"results/"+data_title+'_'+e_method+'_out_'+params+'.csv'
        file = open(res_filename, "r")
        lines = reader(file)
        dataset = list(lines)
        for row in dataset:
            for col in range(len(dataset[0])):
                row[col] = float(row[col].strip())
        dataset = np.array(dataset)
        dataset = np.transpose(dataset)
        
        out_z = np.array(dataset[e_method])[:,j] - np.array(dataset[e_method])[:,len(methods)-1]
        out_z2 = pow(np.array(dataset[e_method])[:,j] - np.array(dataset[e_method])[:,len(methods)-1], 2)       
        out_me = np.mean(out_z)
        out_se = np.sqrt(np.sum(out_z2)/(total_reps-1)-(total_reps/(total_reps-1))*pow(out_me,2))/np.sqrt(total_reps)
        out_norm = out_z / out_se
        out_rep[data_title] += [out_norm]
        
        
#        plt.subplot(1, 2, j+1)
#        plt.title(str(e_method), fontsize=20)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        bp = plt.boxplot(out_norm, positions = list(range(1,len(data_titles)+1)), widths = 0.9)
#            bp = plt.boxplot(dataset.tolist(),positions = [1, 2, 3, 4], widths = 0.9)
        set_box_colors(bp)
        frame1 = plt.gca()  
        frame1.axes.set_xticklabels(data_titles, fontsize=14, rotation = 90)
        
#    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
#    plt.savefig(fig_file+data_title+"_"+e_method+"_"+params+"_new.png".format(1))
        

### all in one figure
##colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
#plt.style.use('seaborn-whitegrid')   
#plt.clf()
#for col, data_title in enumerate(data_titles):    
#    plt.plot(methods[:-1], in_rep1[data_title], '-'+colors[col], label = 'in-sample '+data_title)
#    plt.plot(methods[:-1], out_rep1[data_title], '--'+colors[col], label = 'out-sample')
#plt.title("Normalized Paired Difference Scores of SSE-based Tree and Score-based Tree")
#plt.xlabel("Trees build by")
#plt.legend()
##    
###dummy lines with NO entries, just to create the black style legend
##dummy_lines = []
##dummy_lines.append(plt.plot([],[], '-black')[0])
##dummy_lines.append(plt.plot([],[], '--black')[0])
##legend1 = plt.legend()
##legend2 = plt.legend([dummy_lines[i] for i in [0,1]], ["in-sample", "out-sample"], loc=4)
##plt.add_artist(legend1)
#plt.savefig(directory+"results/PairedDiffNormalized.png".format(1))
#    
#plt.clf()
#for col, data_title in enumerate(data_titles):  
#    plt.plot(methods[:-1], in_rep2[data_title], '-'+colors[col], label = 'in-sample '+data_title)
#    plt.plot(methods[:-1], out_rep2[data_title], '--'+colors[col], label = 'out-sample ')
#plt.title("Percentage of Improved Score Over SSE-based Trees")
#plt.xlabel("Trees build by")
#plt.legend()
#plt.savefig(directory+"results/Percentage.png".format(1))

#
#plt.clf()
#col = 2 
#plt.plot(methods[:-1], in_rep1[data_title], '-'+colors[col], label = 'in-sample '+data_title)
#plt.plot(methods[:-1], out_rep1[data_title], '--'+colors[col], label = 'out-sample')
#plt.title("Normalized Paired Difference Scores of SSE-based Tree and Score-based Tree")
#plt.xlabel("Trees build by")
#plt.legend()
#plt.clf()
#col = 3
#plt.plot(methods[:-1], in_rep2[data_title], '-'+colors[col], label = 'in-sample '+data_title)
#plt.plot(methods[:-1], out_rep2[data_title], '--'+colors[col], label = 'out-sample ')
#plt.title("Percentage of Improved Score Over SSE-based Trees")
#plt.xlabel("Trees build by")
#plt.legend()
#plt.savefig(directory+"results/PairedDiffNormalized_"+data_title+".png".format(1))
##    
    
    ## can run this separately also
#    plt.clf()
#    fig = plt.figure()
##    data_title = 'yield'
#    fig.suptitle(data_title, fontsize=15)
#    plt.subplot(1, 3, 1)
#    plt.title('Normalized Paired Difference', fontsize=10)
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    col = 2 
#    plt.plot(methods[:-1], in_rep1[data_title], '-'+colors[col], label = 'in-sample ')
#    plt.plot(methods[:-1], out_rep1[data_title], '--'+colors[col], label = 'out-sample')
#    plt.legend()
#    frame1 = plt.gca()  
#    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)
#    plt.subplot(1, 3, 2)
#    plt.title('% of Improved Scores', fontsize=10)
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    col = 3
#    plt.plot(methods[:-1], in_rep2[data_title], '-'+colors[col], label = 'in-sample ')
#    plt.plot(methods[:-1], out_rep2[data_title], '--'+colors[col], label = 'out-sample ')
#    plt.legend()
#    frame1 = plt.gca()  
#    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)
#    plt.subplot(1, 3, 3)
#    plt.title('Normalized Paired Difference (by SSE)', fontsize=10)
#    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#    col = 4
#    plt.plot(methods[:-1], in_rep1_sse[data_title], '-'+colors[col], label = 'in-sample ')
#    plt.plot(methods[:-1], out_rep1_sse[data_title], '--'+colors[col], label = 'out-sample ')
#    plt.legend()
#    frame1 = plt.gca()  
#    frame1.axes.set_xticklabels(methods, fontsize=14, rotation = 90)
#    fig.subplots_adjust(hspace=0)
#    fig.tight_layout()
#    plt.savefig(directory+"results/"+data_title+".png".format(1))
#    
