import pandas as pd
import numpy as np
from scipy import stats
import os

def compute_pval(which_dat='Test'):
    pval = np.zeros((4,4,5))
    for th_id, th in enumerate(prune_thr_list):
        for metric_id, metric in enumerate(metrics):
            for method_id, method in enumerate(methods):
                df = df_scores[(df_scores['Method'] == method) & (df_scores['Metric'] == metric) & (df_scores['Threshold'] == th)]
                df_met = df_scores[(df_scores['Method'] == metric) & (df_scores['Metric'] == metric) & (df_scores['Threshold'] == th)]
                if method_id == metric_id:
                    pval[metric_id][method_id][th_id] = 1
                else:
                    if which_dat == 'Test':
                        pval[metric_id][method_id][th_id] = stats.ttest_rel(df_met.Test, df.Test, alternative='less').pvalue
                    else:
                        pval[metric_id][method_id][th_id] = stats.ttest_rel(df_met.Train, df.Train, alternative='less').pvalue
    return pval

def compile_pdf(pval, output_filename='synth1.tex', threshold_id=0):
    input_filename = 'python_to_latex/figure_template.tex'
    file = open(input_filename, "r")
    lines = file.readlines()
    
    modified_lines = []
    for l in lines:
        if len(l) > 1:
            if 'SSE' in l.split()[0]:
               l = l.replace('XXXXX', str(np.round(pval[0][0][threshold_id], 7)))
               l = l.replace('YYYYY', str(np.round(pval[0][1][threshold_id], 7)))
               l = l.replace('ZZZZZ', str(np.round(pval[0][2][threshold_id], 7)))
               l = l.replace('MMMMM', str(np.round(pval[0][3][threshold_id], 7)))
            if 'CRPS' in l.split()[0]:
               l = l.replace('XXXXX', str(np.round(pval[1][0][threshold_id], 7)))
               l = l.replace('YYYYY', str(np.round(pval[1][1][threshold_id], 7)))
               l = l.replace('ZZZZZ', str(np.round(pval[1][2][threshold_id], 7)))
               l = l.replace('MMMMM', str(np.round(pval[1][3][threshold_id], 7)))
            if 'DSS' in l.split()[0]:
               l = l.replace('XXXXX', str(np.round(pval[2][0][threshold_id], 7)))
               l = l.replace('YYYYY', str(np.round(pval[2][1][threshold_id], 7)))
               l = l.replace('ZZZZZ', str(np.round(pval[2][2][threshold_id], 7)))
               l = l.replace('MMMMM', str(np.round(pval[2][3][threshold_id], 7)))
            if 'IS1' in l.split()[0]:
               l = l.replace('XXXXX', str(np.round(pval[3][0][threshold_id], 7)))
               l = l.replace('YYYYY', str(np.round(pval[3][1][threshold_id], 7)))
               l = l.replace('ZZZZZ', str(np.round(pval[3][2][threshold_id], 7)))
               l = l.replace('MMMMM', str(np.round(pval[3][3][threshold_id], 7)))
        modified_lines.append(l)
    
    f = open("python_to_latex/" + output_filename, "a")
    f.writelines(modified_lines)
    f.close()
    
    os.system("/Library/TeX/texbin/pdflatex -output-directory " + "python_to_latex/" + " -no-shell-escape " + "python_to_latex/" + output_filename)

df_scores = pd.read_csv('synth_fig/df1.csv')

metrics = ['sse', 'crps', 'dss', 'is1']
methods = ['sse', 'crps', 'dss', 'is1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]

total_reps = 30
liste = []

pval_train = compute_pval(which_dat='Train')
pval_test = compute_pval(which_dat='Test')

# Choose threshold_id=0, 1, 2, 3, 4
compile_pdf(pval_train, output_filename='tr_synth1_0.tex', threshold_id=0)
compile_pdf(pval_train, output_filename='tr_synth1_1.tex', threshold_id=1)
compile_pdf(pval_train, output_filename='tr_synth1_2.tex', threshold_id=2)
compile_pdf(pval_train, output_filename='tr_synth1_3.tex', threshold_id=3)
compile_pdf(pval_train, output_filename='tr_synth1_4.tex', threshold_id=4)

# Choose threshold_id=0, 1, 2, 3, 4
compile_pdf(pval_test, output_filename='test_synth1_0.tex', threshold_id=0)
compile_pdf(pval_test, output_filename='test_synth1_1.tex', threshold_id=1)
compile_pdf(pval_test, output_filename='test_synth1_2.tex', threshold_id=2)
compile_pdf(pval_test, output_filename='test_synth1_3.tex', threshold_id=3)
compile_pdf(pval_test, output_filename='test_synth1_4.tex', threshold_id=4)
