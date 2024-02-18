import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def tables(df_scores):
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
    
    return table1

def plot_Figure4(df_scores, repno, figlab, is_hard=False):
    methods = ['crps', 'dss', 'is1']
    titles = ['CRPS', 'DSS', 'IS1']
    prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
    nlist = [200, 400, 800, 1600]
    built = 'sse'
    ft = 20
    table1compare = []
    fig, ax = plt.subplots(3, 1, figsize=(4, 12))

    if is_hard == False:
        fig.text(-0.25, 0.5, 'Eval scoring rules', va='center', rotation='vertical', fontsize=ft)

    for mid, evalu in enumerate(methods):
        xmax = 0
        if evalu != built:
            frac = []
            #print('eval:', evalu)
            for nid, ns in enumerate(nlist):
                #print('n:', ns)
                Test_score = []
                for r in prune_thr_list:
                    df1 = df_scores[(df_scores['Method'] == evalu) & (df_scores['Metric'] == evalu) & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
                    Test_score.append({'Score': np.mean(df1['Test']), 'Method': evalu, 'Threshold': r})
      
                    df2 = df_scores[(df_scores['Method'] == built) & (df_scores['Metric'] == built) & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
                    Test_score.append({'Score': np.mean(df2['Test']), 'Method': built, 'Threshold': r})
                    
                dt = pd.DataFrame(Test_score)
                idm1 = dt[dt['Method'] == evalu]['Score'].idxmin()
                idm2 = dt[dt['Method'] == built]['Score'].idxmin()
                
                r1star = dt.iloc[idm1]['Threshold']
                r2star = dt.iloc[idm2]['Threshold']
                
                tab1 = {'Method': evalu, 'Threshold': r1star, 'n': ns}
                table1compare.append(tab1)
                tab1 = {'Method': built, 'Threshold': r2star, 'n': ns}
                table1compare.append(tab1)
                
                df1 = df_scores[(df_scores['Method'] == evalu) & (df_scores['Metric'] == evalu) & (df_scores['Threshold'] == r1star) & (df_scores['n'] == ns)]
                df2 = df_scores[(df_scores['Method'] == built) & (df_scores['Metric'] == evalu) & (df_scores['Threshold'] == r2star) & (df_scores['n'] == ns)]
 
                diff = np.array(df1['Test']) - np.array(df2['Test'])
                    
                frac.append(np.round(100*np.mean(np.array(df1['Test']) <= np.array(df2['Test']))))
                ax[mid].scatter((np.mean(diff)), [nid], color='blue')
                ax[mid].errorbar(y=[nid], x=(np.mean(diff)),
                                    xerr=[1.96*np.std(diff)/np.sqrt(repno)],
                                    capsize=4, color='blue') #label='both limits (default)')

                ax[mid].text(np.mean(diff)-np.std(diff)/np.sqrt(repno), nid + 0.1, str(int(frac[nid])) + '%', fontsize=ft)
                
                if xmax < np.abs(np.mean(diff)) + np.abs(1.96*np.std(diff)/np.sqrt(repno)):
                    xmax = np.abs(np.mean(diff)) + np.abs(1.96*np.std(diff)/np.sqrt(repno))
                
            ax[mid].axvline(0, color='black', linewidth=5)
            ax[mid].set_ylim(-0.5, 3.5)
            ax[mid].set_xlim(-xmax, xmax)
            ax[mid].set_yticks([])
        else:
            ax[mid].spines['top'].set_visible(False)
            ax[mid].spines['bottom'].set_visible(False)
            ax[mid].spines['right'].set_visible(False)
            ax[mid].spines['left'].set_visible(False)
            ax[mid].tick_params(axis='x', colors='white')
            ax[mid].tick_params(axis='y', colors='white')

    if is_hard == False:
        ax[0].set_yticks([0,1,2,3])
        ax[1].set_yticks([0,1,2,3])
        ax[2].set_yticks([0,1,2,3])

        ax[0].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)
        ax[1].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)
        ax[2].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)

    if is_hard == False:
        ax[0].set_ylabel('CRPS', fontsize=ft)
        ax[1].set_ylabel('DSS', fontsize=ft)
        ax[2].set_ylabel('IS1', fontsize=ft)


    if is_hard == False:
        plt.title('Easy Dataset', x=0.5, y=3.5, fontsize=ft)
    else:
        plt.title('Hard Dataset', x=0.5, y=3.5, fontsize=ft)
        
    plt.savefig(figlab, bbox_inches="tight")
    plt.close()

def plot_Figure5(df_scores, repno, figlab, is_hard=False):
    methods = ['crps', 'dss', 'is1', 'sse']
    bmethods = ['crps', 'dss', 'is1']
    titles = ['CRPS', 'DSS', 'IS1', 'SSE']
    prune_thr_list = [0, 0.1, 0.3, 0.5, 0.8]
    nlist = [200, 400, 800, 1600]

    ft = 20
    table1compare = []
    fig, ax = plt.subplots(4, 3, figsize=(9, 12))

    fig.suptitle('Build scoring rules', y=0.06, fontsize=ft)

    if is_hard == False:
        fig.text(-0.05, 0.5, 'Eval scoring rules', va='center', rotation='vertical', fontsize=ft)
    for bid, built in enumerate(bmethods):
        #print('built:', built)
        for mid, evalu in enumerate(methods):
            xmax = 0
            if evalu != built:
                frac = []
                #print('eval:', evalu)
                for nid, ns in enumerate(nlist):
                    #print('n:', ns)
                    Test_score = []
                    for r in prune_thr_list:
                        df1 = df_scores[(df_scores['Method'] == evalu) & (df_scores['Metric'] == evalu) & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
                        Test_score.append({'Score': np.mean(df1['Test']), 'Method': evalu, 'Threshold': r})
          
                        df2 = df_scores[(df_scores['Method'] == built) & (df_scores['Metric'] == built) & (df_scores['Threshold'] == r) & (df_scores['n'] == ns)]
                        Test_score.append({'Score': np.mean(df2['Test']), 'Method': built, 'Threshold': r})
                        
                    dt = pd.DataFrame(Test_score)
                    idm1 = dt[dt['Method'] == evalu]['Score'].idxmin()
                    idm2 = dt[dt['Method'] == built]['Score'].idxmin()
                    
                    r1star = dt.iloc[idm1]['Threshold']
                    r2star = dt.iloc[idm2]['Threshold']
                    
                    tab1 = {'Method': evalu, 'Threshold': r1star, 'n': ns}
                    table1compare.append(tab1)
                    tab1 = {'Method': built, 'Threshold': r2star, 'n': ns}
                    table1compare.append(tab1)
                    
                    df1 = df_scores[(df_scores['Method'] == evalu) & (df_scores['Metric'] == evalu) & (df_scores['Threshold'] == r1star) & (df_scores['n'] == ns)]
                    df2 = df_scores[(df_scores['Method'] == built) & (df_scores['Metric'] == evalu) & (df_scores['Threshold'] == r2star) & (df_scores['n'] == ns)]
     
                    diff = np.array(df1['Test']) - np.array(df2['Test'])
                
                        
                    frac.append(np.round(100*np.mean(np.array(df1['Test']) <= np.array(df2['Test']))))
                    ax[mid, bid].scatter((np.mean(diff)), [nid], color='blue')
                    ax[mid, bid].errorbar(y=[nid], x=(np.mean(diff)),
                                        xerr=[1.96*np.std(diff)/np.sqrt(repno)],
                                        capsize=4, color='blue') #label='both limits (default)')

                    #ax[mid, bid].boxplot(diff, 0, '', 0)
                    ax[mid, bid].text(np.mean(diff)-np.std(diff)/np.sqrt(repno), nid + 0.1, str(int(frac[nid])) + '%', fontsize=ft)
                    
                    if xmax < np.abs(np.mean(diff)) + np.abs(1.96*np.std(diff)/np.sqrt(repno)):
                        xmax = np.abs(np.mean(diff)) + np.abs(1.96*np.std(diff)/np.sqrt(repno))
                    
                ax[mid, bid].axvline(0, color='black', linewidth=5)
                ax[mid, bid].set_ylim(-0.5, 3.5)
                ax[mid, bid].set_xlim(-xmax, xmax)
                ax[mid, bid].set_yticks([])
            else:
                ax[mid, bid].spines['top'].set_visible(False)
                ax[mid, bid].spines['bottom'].set_visible(False)
                ax[mid, bid].spines['right'].set_visible(False)
                ax[mid, bid].spines['left'].set_visible(False)
                ax[mid, bid].tick_params(axis='x', colors='white')
                ax[mid, bid].tick_params(axis='y', colors='white')

    if is_hard == False:
        ax[0, 0].set_yticks([0,1,2,3])
        ax[1, 0].set_yticks([0,1,2,3])
        ax[2, 0].set_yticks([0,1,2,3])
        ax[3, 0].set_yticks([0,1,2,3])
        ax[0, 0].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)
        ax[1, 0].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)
        ax[2, 0].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)
        ax[3, 0].set_yticklabels(['200', '400', '800', '1600'], fontsize=14)
    if is_hard == False:
        ax[0, 0].set_ylabel('CRPS', fontsize=ft)
        ax[1, 0].set_ylabel('DSS', fontsize=ft)
        ax[2, 0].set_ylabel('IS1', fontsize=ft)
        ax[3, 0].set_ylabel('SSE', fontsize=ft)
    ax[3, 0].set_xlabel('CRPS', fontsize=ft)
    ax[3, 1].set_xlabel('DSS', fontsize=ft)
    ax[3, 2].set_xlabel('IS1', fontsize=ft)

    if is_hard == False:
        plt.title('Easy Dataset', x=-0.75, y=4.75, fontsize=ft)
    else:
        plt.title('Hard Dataset', x=-0.75, y=4.75, fontsize=ft)
        
    plt.savefig(figlab, bbox_inches="tight")
    plt.close()
