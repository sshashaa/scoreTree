#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:59:15 2022

@author: ozgesurer
"""


import pandas as pd
import numpy as np
from scipy import stats
import os

def compute_pval(n, prune_thr_list, metrics, methods, df_scores, which_dat='Test'):
    pval = np.zeros((len(metrics), len(methods), len(prune_thr_list), len(n)))
    for n_id, ns in enumerate(n):
        for th_id, th in enumerate(prune_thr_list):
            for metric_id, metric in enumerate(metrics):
                for method_id, method in enumerate(methods):
                    df = df_scores[(df_scores['Method'] == method) & (df_scores['Metric'] == metric) & (df_scores['Threshold'] == th) & (df_scores['n'] == ns)]
                    df_met = df_scores[(df_scores['Method'] == metric) & (df_scores['Metric'] == metric) & (df_scores['Threshold'] == th) & (df_scores['n'] == ns)]
                    if method_id == metric_id:
                        pval[metric_id][method_id][th_id][n_id] = 1
                    else:
                        if which_dat == 'Test':
                            #if (metric_id == 1) & (method_id == 0) & (th_id == 2) & (n_id == 1):
                            #    print(df_met.Test)
                            #    print(df.Test)
                            #    print(stats.ttest_rel(df_met.Test, df.Test, alternative='greater').pvalue)
                            pval[metric_id][method_id][th_id][n_id] = stats.ttest_rel(df_met.Test, df.Test, alternative='greater').pvalue
                        else:
                            pval[metric_id][method_id][th_id][n_id] = stats.ttest_rel(df_met.Train, df.Train, alternative='greater').pvalue
    return pval

def compile_pdf(pval, pvaltest, output_filename='synth1.tex'):
    input_filename = 'python_to_latex/figure_template_compact.tex'
    file = open(input_filename, "r")
    lines = file.readlines()
    
    ssecounter, crpscounter, dsscounter, is1counter = 0, 0, 0, 0
    df_test = np.copy(pvaltest)
    df_tr = np.copy(pval)
    modified_lines = []
    for l in lines:
        if len(l) > 1:
            if 'SSE' in l.split()[0]:
                if ssecounter >= 4:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_tr[0][0][i][ssecounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_tr[0][1][i][ssecounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_tr[0][2][i][ssecounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_tr[0][3][i][ssecounter-4], 2)), 1)
                        
                else:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_test[0][0][i][ssecounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_test[0][1][i][ssecounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_test[0][2][i][ssecounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_test[0][3][i][ssecounter], 2)), 1)

                ssecounter += 1

                    
            if 'CRPS' in l.split()[0]:
                if crpscounter >= 4:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_tr[1][0][i][crpscounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_tr[1][1][i][crpscounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_tr[1][2][i][crpscounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_tr[1][3][i][crpscounter-4], 2)), 1)
                        
                else:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_test[1][0][i][crpscounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_test[1][1][i][crpscounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_test[1][2][i][crpscounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_test[1][3][i][crpscounter], 2)), 1)

                crpscounter += 1


            if 'DSS' in l.split()[0]:
                if dsscounter >= 4:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_tr[2][0][i][dsscounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_tr[2][1][i][dsscounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_tr[2][2][i][dsscounter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_tr[2][3][i][dsscounter-4], 2)), 1)
                else:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_test[2][0][i][dsscounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_test[2][1][i][dsscounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_test[2][2][i][dsscounter], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_test[2][3][i][dsscounter], 2)), 1)

                dsscounter += 1

            if 'IS1' in l.split()[0]:
                if is1counter >= 4:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_tr[3][0][i][is1counter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_tr[3][1][i][is1counter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_tr[3][2][i][is1counter-4], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_tr[3][3][i][is1counter-4], 2)), 1)
                else:
                    for i in range(5):
                        l = l.replace('XXX', str(np.round(df_test[3][0][i][is1counter], 2)), 1)
                    for i in range(5):
                        l = l.replace('YYY', str(np.round(df_test[3][1][i][is1counter], 2)), 1)
                    for i in range(5):
                        l = l.replace('ZZZ', str(np.round(df_test[3][2][i][is1counter], 2)), 1)
                    for i in range(5):
                        l = l.replace('MMM', str(np.round(df_test[3][3][i][is1counter], 2)), 1)

                is1counter += 1


        modified_lines.append(l)
    
    f = open("python_to_latex/" + output_filename, "a")
    f.writelines(modified_lines)
    f.close()
    
    os.system("/Library/TeX/texbin/pdflatex -output-directory " + "python_to_latex/" + " -no-shell-escape " + "python_to_latex/" + output_filename)

n = [200, 400, 800, 1600]
metrics = ['sse', 'crps', 'dss', 'is1']
methods = ['sse', 'crps', 'dss', 'is1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
total_reps = 30
synth_id = [1,2,3,4,5,6,7]


for i in synth_id:
    df_compact = pd.DataFrame()
    for ns in n:
        filename = 'synth_fig/synth' + str(i) + '_minnode_10_n_' + str(ns) + '.csv'
        
        # Read csv.
        df_scores = pd.read_csv(filename)
        df_scores['n'] = ns
        df_compact = pd.concat([df_compact, df_scores])
   
    pval_train = compute_pval(n, prune_thr_list, metrics, methods, df_compact, which_dat='Train')
    pval_test = compute_pval(n, prune_thr_list, metrics, methods, df_compact, which_dat='Test')
    output_filename = 'synth_' + str(i) + '.tex'
    compile_pdf(pval_train, pval_test, output_filename=output_filename)

