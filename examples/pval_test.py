import pandas as pd
import numpy as np
from scipy import stats
import os

def compute_pval(which_dat='Test'):
    pval = np.zeros((4,4,5))
    for th_id, th in enumerate(prune_thr_list):
        #print('threshold:', th)
        for metric_id, metric in enumerate(metrics):
            for method_id, method in enumerate(methods):
                #print('metric:', metric)
                #print('method:', method)
                df = df_scores[(df_scores['Method'] == method) & (df_scores['Metric'] == metric) & (df_scores['Threshold'] == th)]
                df_met = df_scores[(df_scores['Method'] == metric) & (df_scores['Metric'] == metric) & (df_scores['Threshold'] == th)]
                if method_id == metric_id:
                    pval[metric_id][method_id][th_id] = 1
                else:
                    if which_dat == 'Test':
                        pval[metric_id][method_id][th_id] = stats.ttest_rel(df_met.Test, df.Test, alternative='greater').pvalue
                    else:
                        pval[metric_id][method_id][th_id] = stats.ttest_rel(df_met.Train, df.Train, alternative='greater').pvalue
    return pval

def compile_pdf(pval, pvaltest, output_filename='synth1.tex'):
    input_filename = 'python_to_latex/figure_template_thr.tex'
    file = open(input_filename, "r")
    lines = file.readlines()
    
    ssecounter, crpscounter, dsscounter, is1counter = 0, 0, 0, 0
    df = np.copy(pvaltest)
    modified_lines = []
    for l in lines:
        if len(l) > 1:
            if 'SSE' in l.split()[0]:
                l = l.replace('XXXXX', str(np.round(df[0][0][ssecounter], 4)))
                l = l.replace('YYYYY', str(np.round(df[0][1][ssecounter], 4)))
                l = l.replace('ZZZZZ', str(np.round(df[0][2][ssecounter], 4)))
                l = l.replace('MMMMM', str(np.round(df[0][3][ssecounter], 4)))
                ssecounter += 1
                #print(ssecounter)
                if ssecounter >= 5:
                    ssecounter = 0
                    df = np.copy(pval)
            if 'CRPS' in l.split()[0]:
                l = l.replace('XXXXX', str(np.round(df[1][0][crpscounter], 4)))
                l = l.replace('YYYYY', str(np.round(df[1][1][crpscounter], 4)))
                l = l.replace('ZZZZZ', str(np.round(df[1][2][crpscounter], 4)))
                l = l.replace('MMMMM', str(np.round(df[1][3][crpscounter], 4)))
                crpscounter += 1
                #print(crpscounter)
                if crpscounter >= 5:
                    crpscounter = 0
            if 'DSS' in l.split()[0]:
               l = l.replace('XXXXX', str(np.round(df[2][0][dsscounter], 4)))
               l = l.replace('YYYYY', str(np.round(df[2][1][dsscounter], 4)))
               l = l.replace('ZZZZZ', str(np.round(df[2][2][dsscounter], 4)))
               l = l.replace('MMMMM', str(np.round(df[2][3][dsscounter], 4)))
               dsscounter += 1
               #print(dsscounter)
               if dsscounter >= 5:
                   dsscounter = 0
            if 'IS1' in l.split()[0]:
               l = l.replace('XXXXX', str(np.round(df[3][0][is1counter], 4)))
               l = l.replace('YYYYY', str(np.round(df[3][1][is1counter], 4)))
               l = l.replace('ZZZZZ', str(np.round(df[3][2][is1counter], 4)))
               l = l.replace('MMMMM', str(np.round(df[3][3][is1counter], 4)))
               is1counter += 1
               #print(is1counter)
               if is1counter >= 5:
                   is1counter = 0
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
synth_id = [1, 2, 3, 4, 5, 6, 7]

for i in synth_id:
    for ns in n:
        filename = 'synth_fig/synth' + str(i) + '_minnode_10_n_' + str(ns) + '.csv'
        output_filename = 'synth_' + str(i) + '_n_' + str(ns) + '.tex'
        
        # Read csv.
        df_scores = pd.read_csv(filename)
        
        liste = []
        pval_train = compute_pval(which_dat='Train')
        pval_test = compute_pval(which_dat='Test')
    
        compile_pdf(pval_train, pval_test, output_filename=output_filename)

