#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:55:40 2024

@author: ozgesurer
"""

import pandas as pd
from utils import plot_papercombinedfig, plot_box
dfeasy = []
total_reps = 100
for n in [200, 400, 800, 1600]:
    for repno in range(total_reps):
        filename = 'less_noise_examples_100/' + 'res_' + 'synth6_rep_' + str(n) + '_' + str(repno) +'.csv' + '.csv'
        df_saved = pd.read_csv(filename, header=0)
        dfeasy.append(df_saved)

dfeasy = pd.concat(dfeasy) 

plot_papercombinedfig(dfeasy, repno=total_reps, is_hard=False)    
plot_box(dfeasy, repno=total_reps, is_hard=False)

from utils import plot_papercombinedfig, plot_box
dfeasy = []
total_reps = 100
for n in [200, 400, 800, 1600]:
    for repno in range(total_reps):
        filename = 'less_noise_examples_100/' + 'res_' + 'synth7_rep_' + str(n) + '_' + str(repno) +'.csv' + '.csv'
        df_saved = pd.read_csv(filename, header=0)
        dfeasy.append(df_saved)

dfeasy = pd.concat(dfeasy) 

plot_papercombinedfig(dfeasy, repno=total_reps, is_hard=True)    
plot_box(dfeasy, repno=total_reps, is_hard=True)
    

# import pandas as pd
# from utils import plot_papercombinedfig, plot_box
# dfeasy = []
# total_reps = 2
# for n in [1600]:
#     for repno in range(total_reps):
#         filename = 'less_noise_examples_100/' + 'synth7_rep_' + str(repno) +'.csv'
#         df_saved = pd.read_csv(filename, header=None)
#         plt.hist(df_saved[0])
#         plt.show()
#         plt.hist(df_saved[1])
#         plt.show()
