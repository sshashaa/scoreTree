from scoreCART.scoreCARTprune import scoreCART
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import Parallel, delayed
import random


def column(matrix, i):
    return [row[i] for row in matrix]

# inputs
max_depth = 4
min_node_size = 10
num_quantiles = 20
total_reps = 30 
alpha = .2
tol = 0


def OneRep(k, n):
    # random.seed(k + 10)
    filename = 'less_noise_examples/synth6_rep_' + str(k) + '.csv'
    rows = pd.read_csv(filename, header=None)
    rows = rows.values.tolist()[0:n]
    is_cat = []
    cov_uniqvals = []
    x_dim = len(rows[0])-1
    for i in range(x_dim):
        unique_vals = list(sorted(set(column(rows, i))))
        cov_uniqvals += [unique_vals]
        if len(unique_vals) <= 2:
            is_cat += [1]
        else:
            is_cat += [0]

    filename = 'less_noise_examples/synth6_test' + '.csv'
    rows_test = pd.read_csv(filename, header=None)
    rows_test = rows_test.values.tolist()

    dataset = [rows, rows_test]
    
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
    
    actual = [row[-1] for row in dataset[1]]
    actual_in = [row[-1] for row in dataset[0]]
    
    methods = ["crps", "dss", "is1", "sse"]
    dictable = []
    prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
    for pr in prune_thr_list:
        for m in methods: 
            # Fit the tree model
            CARTmodel = scoreCART(m, 
                                  rows, 
                                  tol, 
                                  max_depth, 
                                  min_node_size, 
                                  num_quantiles, 
                                  alpha,
                                  pr,
                                  args = {'is_cat': is_cat, 'cov_uniqvals': cov_uniqvals})
            CARTmodel.build_tree()
            # fittedtree = CARTmodel.fittedtree
            dict_eval = CARTmodel.accuracy_val(test_set, 
                                               actual, 
                                               self_test, 
                                               actual_in, 
                                               metrics=['sse', 'crps', 'dss', 'is1'])
            for metr in ['sse', 'crps', 'dss', 'is1']:
                d = {'Method': m, 'Metric': metr, 'Train': np.round(dict_eval[metr][1], 2), 'Test': np.round(dict_eval[metr][0], 2), 'Threshold': pr, 'n': n}
                dictable.append(d)
    dfres = pd.DataFrame(dictable)
    return dfres

score_list = []
for n in [200, 400, 800, 1600]:
    # Run parallel for each replicate
    scores_reps = Parallel(n_jobs=min(total_reps, 20))(delayed(OneRep)(rep_no, n) for rep_no in range(total_reps))   
    score_list.append(scores_reps)

liste = []    
for nid in range(0, 4):
    scores_reps = score_list[nid]
    for i in range(total_reps):
        ls = scores_reps[i].copy()
        ls['Rep'] = i
        liste.append(ls)
df_scores = pd.concat(liste)

n = [200, 400, 800, 1600]
methods = ['crps', 'dss', 'is1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plt.rcParams['xtick.labelsize'] = 14 
for mid, m in enumerate(methods):
    axes[mid].set_title(m, fontsize=14)
    frac = []
    for nid, ns in enumerate(n):
        #filename = 'less_noise_examples/synth' + str(i) + '_n_' + str(ns) + '.csv'
        #df_scores = pd.read_csv(filename)
        
        Test_score = []
        
        for r in prune_thr_list:
            
            df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
            Test_score.append({'Score': np.mean(df1['Test']), 'Method': m, 'Threshold': r})
            
            df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == 'sse') & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
            Test_score.append({'Score': np.mean(df2['Test']), 'Method': 'sse', 'Threshold': r})
    
        dt = pd.DataFrame(Test_score)

        print(dt)
        idm = np.argmin(dt[dt['Method'] == m]['Score'])
        idsse = np.argmin(dt[dt['Method'] == 'sse']['Score'])
        
        idm = dt[dt['Method'] == m]['Score'].idxmin()
        idsse = dt[dt['Method'] == 'sse']['Score'].idxmin()
        
    
        print('n:', ns)
        print('Method:', m)
        print(idm)
        print('Thr:', dt.iloc[idm]['Threshold'])

        print('Method:', 'sse')
        print(idsse)
        print('Thr:', dt.iloc[idsse]['Threshold'])
                
        df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == dt.iloc[idm]['Threshold']) & (df_scores['n'] == ns)]
        df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == dt.iloc[idsse]['Threshold']) & (df_scores['n'] == ns)]

        frac.append(str(100*np.round(np.mean(np.array(df1['Test']) <= np.array(df2['Test'])), 2)))
        
        
        diff = np.array(df1['Test']) - np.array(df2['Test'])

        axes[mid].scatter((np.mean(np.array(df1['Test']) - np.array(df2['Test']))), [nid])
        axes[mid].errorbar(y=[nid], x=(np.mean(diff)), 
                           xerr=[1.96*np.std(diff)/np.sqrt(30)],
                           capsize=4, label='both limits (default)')
        
        axes[mid].text(np.mean(diff)-np.std(diff)/np.sqrt(30), nid + 0.1, str(frac[nid]) + '%', fontsize=14)
    axes[mid].axvline(0, color='black', linewidth=2)


axes[0].set_yticks([0,1,2,3])
axes[0].set_yticklabels(['200', '400', '800', '1600'], fontsize=16)
axes[0].set_ylabel('n', fontsize=16)
axes[1].set_yticks([])
axes[2].set_yticks([])
axes[0].set_ylim(-0.5, 3.5)
axes[1].set_ylim(-0.5, 3.5)
axes[2].set_ylim(-0.5, 3.5)
plt.show()

