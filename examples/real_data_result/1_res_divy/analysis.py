import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

def readdata(n):
    df = []
    for i in range(30):
        filename = str(n) + '_node500new' + '/' + 'rep' + str(i) + '.txt'
        with open(filename) as f:
            lines = f.readlines()
        
        
        for l in lines:
            l = l.replace("[", "")
            l = l.replace("]", "")
            l = l.replace("'", "")
            d = {'Method': str(l.split()[0]), 'Metric': str(l.split()[1]), 'Train': float(l.split()[2]), 'Test': float(l.split()[3]), 'Threshold': float(l.split()[4]), 'rep': i}
            df.append(d)
            
    df_scores = pd.DataFrame(df)
    
    return df_scores




n = [5000, 10000]
methods = ['crpsnew', 'dss', 'is1']
titles = ['CRPS', 'DSS', 'IS1']
prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plt.rcParams['xtick.labelsize'] = 14 
for mid, m in enumerate(methods):
    axes[mid].set_title(titles[mid], fontsize=14)
    frac = []
    for nid, ns in enumerate(n):
        
        df_scores = readdata(ns)
        print(df_scores)
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
        print('Thr:', dt.iloc[idm]['Threshold'])

        print('Method:', 'sse')
        print('Thr:', dt.iloc[idsse]['Threshold'])
                
        df1 = df_scores[(df_scores['Method'] == m) & (df_scores['Metric'] == m) & (df_scores['Threshold'] == dt.iloc[idm]['Threshold'])]
        df2 = df_scores[(df_scores['Method'] == 'sse') & (df_scores['Metric'] == m) & (df_scores['Threshold'] == dt.iloc[idsse]['Threshold'])]
        
        frac.append(str(np.round(100*np.mean(np.array(df1['Test']) <= np.array(df2['Test'])), 2)))
        #print(frac)
        
        diff = np.array(df1['Test']) - np.array(df2['Test'])
        # print(len(diff))
        axes[mid].scatter((np.mean(np.array(df1['Test']) - np.array(df2['Test']))), [nid])
        axes[mid].errorbar(y=[nid], x=(np.mean(diff)), 
                           xerr=[1.96*np.std(diff)/np.sqrt(30)],
                           capsize=4, label='both limits (default)')
        
        axes[mid].text(np.mean(diff)-np.std(diff)/np.sqrt(30), nid + 0.1, str(frac[nid]) + '%', fontsize=14)
    axes[mid].axvline(0, color='black', linewidth=2)


axes[0].set_yticks([0,1])
axes[0].set_yticklabels(['5000', '10000'], fontsize=16)
axes[0].set_ylabel('n', fontsize=16)
axes[1].set_yticks([])
axes[2].set_yticks([])
axes[0].set_ylim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)
axes[2].set_ylim(-0.5, 1.5)
plt.savefig("Figure5b.png", bbox_inches="tight")
plt.close()
