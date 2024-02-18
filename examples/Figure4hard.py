from scoreTree.scoreTreeprune import scoreTree
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
import random
from utils import tables

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
    filename = 'less_noise_examples/synth7_rep_' + str(k) + '.csv'
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

    filename = 'less_noise_examples/synth7_test' + '.csv'
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
            CARTmodel = scoreTree(m, 
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

# Generage Figure 3
table1 = tables(df_scores)

# Generate Figure 2
figure2 = df_scores[(df_scores['Method'].isin(['crps', 'sse'])) & (df_scores['Metric'] == 'crps') & (df_scores['Threshold'].isin([0, 0.5]))]
figure2sse = df_scores[(df_scores['Method'].isin(['sse'])) & (df_scores['Metric'] == 'crps') & (df_scores['Threshold'].isin([0, 0.5]))]
figure2crps = df_scores[(df_scores['Method'].isin(['crps'])) & (df_scores['Metric'] == 'crps') & (df_scores['Threshold'].isin([0, 0.5]))]
ft = 20
fig2sse = figure2sse[['Test', 'Train', 'Threshold', 'n', 'Rep']]
fig2crps = figure2crps[['Test', 'Train', 'Threshold', 'n', 'Rep']]
fig2crps['Test'] = np.array(fig2crps['Test']) - np.array(fig2sse['Test']) 

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig2 = sns.boxplot(x='n', y='Test', 
            hue='Threshold', palette=sns.color_palette(('blue', 'red')), boxprops=dict(alpha=.7),
            data=fig2crps)
fig2.axhline(y=0, color='black', linestyle='--', linewidth=3)
fig2.set_title('CRPS(CRPS tree) - CRPS(SSE tree)', fontsize=ft)
fig2.set(ylabel=None)
fig2.set_xlabel('data size', fontsize=ft)
fig2.tick_params(axis='both', which='major', labelsize=ft)
fig2.legend(title='Pruning threshold', fontsize=ft)
plt.setp(ax.get_legend().get_title(), fontsize=ft) 
plt.savefig("Figures/Figure2.png", bbox_inches="tight")
plt.close()

# Generate Table 2
table1 = pd.DataFrame(table1)
methods = ['sse', 'crps', 'dss', 'is1']
print("Table 2: The optimal pruning for each scoring rule and size (hard dataset)")
print("     " + str(nlist))
for m in methods:
    vals = np.array(table1[(table1['Method'] == m)]['Threshold'])
    print(m + "   " , end="")
    for v in vals:
        print(str(v) + "  ", end="") 
    print("\n")
    
from utils import plot_Figure4, plot_Figure5
plot_Figure4(df_scores, repno=total_reps, figlab='Figure4hard.png', is_hard=True)
plot_Figure5(df_scores, repno=total_reps, figlab='Figure5hard.png', is_hard=True)