#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:46:35 2022

@author: ozgesurer
"""
import numpy as np
from newsplit import sse_for_new_split, crps_for_new_split, dss_for_new_split, is1_for_new_split

class scoreCART():
    def __init__(self, method, train_set, tol, max_depth, min_node_size, num_quantiles, args):

        self.train_set = train_set
        self.x_dim = len(train_set[0]) - 1
        self.method = method
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.num_quantiles = num_quantiles
        self.args = args
        self.new_split_funcs = eval(method + '_for_new_split')
        self.is_cat = args['is_cat']
        self.cov_uniqvals = args['cov_uniqvals']
        self.tol = tol
                
    # Find the empirical cdf of a sample, Outcome: quantiles and cumulative probabilities
    def ecdf(self, sample):
        sample = np.atleast_1d(sample)
        quantiles, counts = np.unique(sample, return_counts=True)
        cumprob = np.cumsum(counts).astype(np.double) / sample.size
        return quantiles, cumprob

    def column(self, matrix, i):
        return [row[i] for row in matrix]

    def test_split(self, index, value, train_set, equal):
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

    def get_split(self, nodetr):
        #train_set = self.train_set 
        x_dim = self.x_dim 
        num_quantiles = self.num_quantiles
        min_node_size = self.min_node_size 
        b_index, b_value, b_groups = 999, 999, None
        b_score = self.new_split_funcs(nodetr, 0)
        first_val = 0  
        split_occurs = 0
        
        for index in range(x_dim):
            qe, pe = self.ecdf(self.column(nodetr, index))
            if self.is_cat[index]:# and len(unique_vals) <= 25:
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
                groups = self.test_split(index, val, nodetr, equal)
                if len(groups[0]) >= min_node_size and len(groups[1]) >= min_node_size:
                    measure =  self.new_split_funcs(groups, 1)
                    if not first_val:
                        first_val = 1
                        if b_score < measure:
                            print("monotonicity violated - " + str(self.method) + " - variable "+str(index))
                            # log_file.write("monotonicity violated - "+ str(self.method) + " - variable "+str(val))
                        b_score = max(b_score, measure)
                    if split_occurs:
                        check_tol = 0
                    else:
                        check_tol = self.tol
    
                    if measure <= b_score*(1 - check_tol):                    
                        split_occurs = 1
                        b_index, b_value, b_score, b_groups = index, val, measure, groups
        if not split_occurs:
            print("no improvement - " + str(self.method))
            # log_file.write("no improvement - " + str(self.method))
        return {'index':b_index, 'value':b_value, 'groups':b_groups}  

    # Return the observaions in the leaf    
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return outcomes
    
    def split(self, node, depth):

        max_depth = self.max_depth
        min_node_size = self.min_node_size 

        if node['groups']:
            left, right = node['groups']
            del(node['groups'])
        else:
            print('NOTHING')
            
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        
        # process left child
        if len(left) < 3*min_node_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth+1)
            
        # process right child
        if len(right) < 3*min_node_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth+1)

    # Print a decision tree
    def print_tree(self, node, depth=0):
        is_cat = self.is_cat
        if isinstance(node, dict):
            if is_cat[node['index']]:
                print('%s[X%d = %d]' % ((depth*' ', (node['index']+1), int(node['value']))))
            else:
                print('%s[X%d < %.4f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', len(node)))) 
                
    def build_tree(self):
        root = self.get_split(self.train_set)
        self.split(root, 1)
        print("tree_method " + self.method + "\n###########################")
        self.print_tree(root, depth=0)
        return root
        
