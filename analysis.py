

import pandas as pd
import numpy as np
from scipy import stats
import os
import seaborn as sns
import matplotlib.pyplot as plt


n = [200, 400, 800, 1600]
methods = ['crps', 'dss', 'is1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
i = 7

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plt.rcParams['xtick.labelsize'] = 14 
for mid, m in enumerate(methods):
    axes[mid].set_title(m, fontsize=14)
    frac = []
    for nid, ns in enumerate(n):
        filename = 'less_noise_examples/synth' + str(i) + '_n_' + str(ns) + '.csv'
        df_scores = pd.read_csv(filename)
        
        Test_score = []
        
        for r in prune_thr_list:
            
            df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == r)]
            Test_score.append({'Score': np.mean(df1['Test']), 'Method': m, 'Threshold': r})
            
            df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == 'sse') & (df_scores['Threshold'] == r)]
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
                
        df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == dt.iloc[idm]['Threshold'])]
        df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == dt.iloc[idsse]['Threshold'])]

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

