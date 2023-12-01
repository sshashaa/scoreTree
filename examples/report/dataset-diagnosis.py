#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:24:26 2022

@author: sarashashaani
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from scipy.stats import moment
from IPython.display import display, HTML

data = []
datasize = 1600
x = np.random.uniform(low=-1, high=1, size=datasize)
data.append(x)

# dataset 1

shape1, scale1 = 3, 3 
shape2, scale2 = 1, 9 
shape3, scale3 = 1, 1
shape4, scale4 = 0, 3 

data1 = []
for id_x, x_i in enumerate(x):
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
    
    data1.append(float(y))
data.append(data1)    
    
# dataset 2

shape1, scale1 = 1, 3 
shape2, scale2 = 1, 5 
shape3, scale3 = 3, 1
shape4, scale4 = 1, 3 

data2 = []
for id_x, x_i in enumerate(x):
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
    
    data2.append(float(y))
data.append(data2)  

# dataset 3

shape1, scale1 = 7, 2
shape2, scale2 = 8, 3 
shape3, scale3 = 9, 5
shape4, scale4 = 1, 10 

data3 = []
for id_x, x_i in enumerate(x):
    if x_i < 0:
        if x_i < -0.5:
            y = np.random.gamma(shape1, scale1, 1)
        else:
            y = np.random.gamma(shape2, scale2, 1)                    
    else:
        if x_i < 0.5:
            y = np.random.gamma(shape3, scale3, 1)
        else:
            y = np.random.gamma(shape4, scale4, 1)                
    
    data3.append(float(y))
data.append(data3)  

# dataset 4

shape1, scale1 = 4, 1
shape2, scale2 = 4, 2 
shape3, scale3 = 4, 3
shape4, scale4 = 4, 4 

data4 = []
for id_x, x_i in enumerate(x):
    if x_i < 0:
        if x_i < -0.5:
            y = np.random.gamma(shape1, scale1, 1)
        else:
            y = np.random.gamma(shape2, scale2, 1)                    
    else:
        if x_i < 0.5:
            y = np.random.gamma(shape3, scale3, 1)
        else:
            y = np.random.gamma(shape4, scale4, 1)                
    
    data4.append(float(y))
data.append(data4)  

# dataset 5

shape1, scale1 = 2, 5
shape2, scale2 = 5, 2 
shape3, scale3 = 1, 5
shape4, scale4 = 5, 1 

data5 = []
for id_x, x_i in enumerate(x):
    if x_i < 0:
        if x_i < -0.5:
            y = np.random.gamma(shape1, scale1, 1)
        else:
            y = np.random.gamma(shape2, scale2, 1)                    
    else:
        if x_i < 0.5:
            y = np.random.gamma(shape3, scale3, 1)
        else:
            y = np.random.gamma(shape4, scale4, 1)                
    
    data5.append(float(y))
data.append(data5)  

# dataset 6

shape1, scale1 = 2, 1/2
shape2, scale2 = 3, 1/3 
shape3, scale3 = 4, 1/4
shape4, scale4 = 5, 1/5 

data6 = []
for id_x, x_i in enumerate(x):
    if x_i < 0:
        if x_i < -0.5:
            y = np.random.lognormal(shape1, scale1, 1)
        else:
            y = np.random.lognormal(shape2, scale2, 1)                    
    else:
        if x_i < 0.5:
            y = np.random.lognormal(shape3, scale3, 1)
        else:
            y = np.random.lognormal(shape4, scale4, 1)                
    
    data6.append(float(y))
data.append(data6)    

# dataset 7

shape1, scale1 = 1/2, 1/2
shape2, scale2 = 1/3, .6 
shape3, scale3 = 1/4, .3
shape4, scale4 = 1/5, .3 

data7 = []
for id_x, x_i in enumerate(x):
    if x_i < 0:
        if x_i < -0.5:
            y = np.random.lognormal(shape1, scale1, 1)
        else:
            y = np.random.lognormal(shape2, scale2, 1)                    
    else:
        if x_i < 0.5:
            y = np.random.lognormal(shape3, scale3, 1)
        else:
            y = np.random.lognormal(shape4, scale4, 1)                
    
    data7.append(float(y))
data.append(data7)  


f = pd.DataFrame(np.transpose(data), columns = ['X','Y1','Y2','Y3','Y4','Y5','Y6','Y7'])
f1 = f.loc[(f['X'] > -1) & (f['X'] < -.5)]
f2 = f.loc[(f['X'] > -.5) & (f['X'] < 0)]
f3 = f.loc[(f['X'] > 0) & (f['X'] < 0.5)]
f4 = f.loc[(f['X'] > .5) & (f['X'] < 1)]

display(f1)
np.around(moment(f1, moment=3),1) 
np.around(moment(f2, moment=3),1) 
np.around(moment(f3, moment=3),1) 
np.around(moment(f4, moment=3),1) 

### plotting
for i in range(7):
    n, bins, patches = plt.hist(data[i+1], bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Dataset '+str(i+1))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    
    plt.hist(data1)
    plt.show()



