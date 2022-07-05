#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:25:21 2022

@author: ozgesurer
"""
import csv
from scoreCARTprune import scoreCART
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import Parallel, delayed
import random
import seaborn as sns

# Training
n = 1600
for i in range(30):
    x = np.random.uniform(low=-1, high=1, size=n)
    filename = 'less_noise_examples/' + 'x' + '_rep_' + str(i) 
    pd.DataFrame(x).to_csv(filename + '.csv', sep=',')

# Test 
for i in range(1):
    x = np.random.uniform(low=-1, high=1, size=1000)
    filename = 'less_noise_examples/' + 'xtest'
    pd.DataFrame(x).to_csv(filename + '.csv', sep=',')

shape1, scale1 = 2, 1/2
shape2, scale2 = 3, 1/3
shape3, scale3 = 4, 1/4
shape4, scale4 = 5, 1/5

def synthetic6(n, repno, is_test=False):
    if is_test:
        filename = 'less_noise_examples/' + 'xtest' + '.csv'
    else:
        filename = 'less_noise_examples/' + 'x' + '_rep_' + str(repno) + '.csv'
        
    # Read csv.
    x = pd.read_csv(filename, header=0)
    # x = np.random.uniform(low=-1, high=1, size=n)
    data = []
    for id_x in range(n):
        x_i = x['0'][id_x]
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


for i in range(30):
    data = synthetic6(n=1600, repno=i)
    filename = 'less_noise_examples/' + 'synth6' + '_rep_' + str(i) + '.csv'
    file = open(filename, 'a+', newline ='')
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(data)
        
data = synthetic6(n=1000, repno=i, is_test=True)
filename = 'less_noise_examples/' + 'synth6' + '_test' + '.csv'
file = open(filename, 'a+', newline ='')
with file:   
    write = csv.writer(file)
    write.writerows(data)
    
    
        
shape1, scale1 = 1/2, 0.5

shape2, scale2 = 1/3, 0.6

shape3, scale3 = 1/4, 0.3

shape4, scale4 = 1/5, 0.3

def synthetic7(n, repno, is_test):
    if is_test:
        filename = 'less_noise_examples/' + 'xtest' + '.csv'
    else:
        filename = 'less_noise_examples/' + 'x' + '_rep_' + str(repno) + '.csv'
        
    # Read csv.
    x = pd.read_csv(filename, header=0)
    
    # x = np.random.uniform(low=-1, high=1, size=n)
    data = []
    for id_x in range(n):
        x_i = x['0'][id_x]
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

for i in range(30):
    data = synthetic6(n=1600, repno=i)
    filename = 'less_noise_examples/' + 'synth7' + '_rep_' + str(i) + '.csv'
    file = open(filename, 'a+', newline ='')
    # writing the data into the file
    with file:   
        write = csv.writer(file)
        write.writerows(data)

data = synthetic6(n=1000, repno=i, is_test=True)
filename = 'less_noise_examples/' + 'synth7' + '_test' + '.csv'
file = open(filename, 'a+', newline ='')
with file:   
    write = csv.writer(file)
    write.writerows(data)
    