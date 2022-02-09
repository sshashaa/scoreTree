#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:49:13 2022

@author: ozgesurer
"""
import ast
import numpy as np
from collections import Counter

def accuracy_sse(leaf_dict, args):
    total_sse = 0
    for key, val in leaf_dict.items():
        leaf = ast.literal_eval(key)
        avg = np.mean(leaf)
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_sse += pow(point - avg, 2)*rv[j]
    return total_sse

def accuracy_crps(leaf_dict, args):  
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


def accuracy_dss(leaf_dict, args):  
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

def accuracy_is1(leaf_dict, args):
    alpha = args['alpha']
    total_is = 0
    for key, val in leaf_dict.items():
        leaf = sorted(ast.literal_eval(key))
        u = leaf[int(np.ceil((1-alpha)*len(leaf)))-1]
        
        xv = list(Counter(val).keys()) # equals to list(set(targets))
        rv = list(Counter(val).values())

        for j, point in enumerate(xv):
            total_is += (u+(point-u)*(point>=u)/alpha)*rv[j]
    return total_is