from scoreCART.scoreCARTprune import scoreCART
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from joblib import Parallel, delayed
import random
from utils import plot_paperfig


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
                d = {'Method': m, 'Metric': metr, 
                     'Train': np.round(dict_eval[metr][1], 2), 
                     'Test': np.round(dict_eval[metr][0], 2), 
                     'Threshold': pr, 'n': n}
                dictable.append(d)
    dfres = pd.DataFrame(dictable)
    return dfres

score_list = []
nlist = [200, 400, 800, 1600]
for n in nlist:
    # Run parallel for each replicate
    scores_reps = Parallel(n_jobs=min(total_reps, 20))(delayed(OneRep)(rep_no, n) for rep_no in range(total_reps))   
    score_list.append(scores_reps)

liste = []    
for nid in range(0, len(nlist)):
    scores_reps = score_list[nid]
    for i in range(total_reps):
        ls = scores_reps[i].copy()
        ls['Rep'] = i
        liste.append(ls)
df_scores = pd.concat(liste)

table1 = plot_paperfig(df_scores, "Figures/Figure3easy.png")

# Generate Table 2
table1 = pd.DataFrame(table1)
methods = ['sse', 'crps', 'dss', 'is1']
print("Table 1: The optimal pruning for each scoring rule and size (hard dataset)")
print("     " + str(nlist))
for m in methods:
    vals = np.array(table1[(table1['Method'] == m)]['Threshold'])
    print(m + "   " , end="")
    for v in vals:
        print(str(v) + "  ", end="") 
    print("\n")
    