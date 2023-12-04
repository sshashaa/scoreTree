import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def plot_paperfig(df_scores, figlab):
    methods = ['crps', 'dss', 'is1']
    titles = ['CRPS', 'DSS', 'IS1']
    prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
    nlist = [200, 400, 800, 1600]
    table1 = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.rcParams['xtick.labelsize'] = 14 
    for mid, m in enumerate(methods):
        axes[mid].set_title(titles[mid], fontsize=14)
        frac = []
        for nid, ns in enumerate(nlist):

            Test_score = []
            
            for r in prune_thr_list:
                
                df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
                Test_score.append({'Score': np.mean(df1['Test']), 'Method': m, 'Threshold': r})
                
                df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == 'sse') & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
                Test_score.append({'Score': np.mean(df2['Test']), 'Method': 'sse', 'Threshold': r})
        
            dt = pd.DataFrame(Test_score)

            # print(dt)
            idm = np.argmin(dt[dt['Method'] == m]['Score'])
            idsse = np.argmin(dt[dt['Method'] == 'sse']['Score'])
            
            idm = dt[dt['Method'] == m]['Score'].idxmin()
            idsse = dt[dt['Method'] == 'sse']['Score'].idxmin()
            
            tab1 = {'Method': m, 'Threshold': dt.iloc[idm]['Threshold'], 'n': ns}
            table1.append(tab1)
            
            if m == 'crps':
                tab1 = {'Method': 'sse', 'Threshold': dt.iloc[idsse]['Threshold'], 'n': ns}
                table1.append(tab1)

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
    plt.savefig(figlab, bbox_inches="tight")
    plt.close()
    
    return table1
