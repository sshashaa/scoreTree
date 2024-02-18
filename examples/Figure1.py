'''
Creates the figure for the toy example in the paper
'''
from scoreTree.scoreTreeprune import scoreTree
from random import sample
import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter
from scipy.stats import norm, expon

def sse_for_new_split(groups):
    sse = 0
    for group in groups:
        mean_target = sum([row[-1] for row in group])/float(len(group))
        sse += sum([pow(row[-1]-mean_target,2) for row in group])
    return sse


def crps_for_new_split(groups):
    total_crps = 0          

    for group in groups:
        targets = sorted(list(np.asarray([row[-1] for row in group])))        
        leaf_ytrain = list(Counter(targets).keys())
        leaf_ytrain_freq = list(Counter(targets).values())

        nlength = len(leaf_ytrain)
        denum   = (2/(nlength*nlength))
        idlist  = np.arange(1, nlength + 1)
        
        total_crps += denum*sum([sum((leaf_ytrain - y) * (nlength * (leaf_ytrain > y) - idlist + 0.5) * leaf_ytrain_freq)*leaf_ytrain_freq[j] for j, y in enumerate(leaf_ytrain)])
    return total_crps

def column(matrix, i):
    return [row[i] for row in matrix]

# 19, 22, 24, 34

for seed in [19]:
    np.random.seed(seed)
    
    def synthetic1(n):
        x = np.random.uniform(low=-1, high=1, size=n)
        data = []
        for id_x, x_i in enumerate(x):
            if x_i <= 0:
                y = np.random.normal(loc=1, scale=2, size=1)
            else:
                y = np.random.exponential(scale=1.0, size=1)
            data.append([x_i, float(y[0])])
        return data
    
    n = 1000
    rows = synthetic1(n)
    x_dim = len(rows[0])-1
    
    maxval = np.max(np.array(rows)[:, 1])
    minval = np.min(np.array(rows)[:, 1])
    
    yaxvals = np.arange(minval, maxval, 0.01)
    rv = norm(loc=1, scale=2)
    pdfnorm = rv.pdf(yaxvals)
    
    rv = expon(loc=0, scale=1)
    pdfexp = rv.pdf(yaxvals)
    #### #### #### #### #### #### ####
    # Creta the figure in the paper #
    #### #### #### #### #### #### ####
    
    slist = np.arange(-99, 99)/100
    sselist = []
    crpslist = []
    for s in slist:
    
        right = []
        left = []
        
        for row in rows:
            if row[0] > s:
                right.append(row)
            else:
                left.append(row)
        groups = []
        groups.append(left)
        groups.append(right)
        
        sse = sse_for_new_split(groups)
        crps = crps_for_new_split(groups)
        
        sselist.append(sse)
        crpslist.append(crps)
    
    ft = 20
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0,0].plot(slist, sselist, color='darkred')
    axs[0,0].set_xlabel(r'$s$', fontsize=ft)
    axs[0,0].set_ylabel(r'SSE($s$)', fontsize=ft)
    axs[0,0].set(xticks=[-1, 0, 1], xticklabels=[-1, 0, 1])
    axs[0,0].xaxis.set_tick_params(labelsize=20)
    axs[0,0].set_yticks([])
    
    axs[1,0].plot(slist, crpslist, color='green')
    axs[1,0].set_xlabel(r'$s$', fontsize=ft)
    axs[1,0].set_ylabel(r'CRPS($s$)', fontsize=ft)
    axs[1,0].set(xticks=[-1, 0, 1], xticklabels=[-1, 0, 1])
    axs[1,0].xaxis.set_tick_params(labelsize=20)
    axs[1,0].set_yticks([])

    
    argmin_sse = np.argmin(sselist)
    argmin_crps = np.argmin(crpslist)
    right = []
    left = []
    
    s_sse = slist[argmin_sse]
    s_crps = slist[argmin_crps]
    for sid, s in enumerate([s_sse, s_crps]):
        for row in rows:
            if row[0] > s:
                right.append(row)
            else:
                left.append(row)
        data1 = np.array(left)[:, 1]
        data2 = np.array(right)[:, 1]
        binwidth = 0.5
        
        if sid == 0:
            col = 'darkred'
        else:
            col = 'green'
        num_bins = 50
    
        axs[sid,1].hist(data1, num_bins, facecolor=col, alpha=1, density=True, ec='white')
        axs[sid,1].plot(yaxvals, pdfnorm, color='black')
        axs[sid,1].set_ylabel('Frequency/density', fontsize=ft)
        axs[sid,1].set_yticks([])
        axs[sid,1].xaxis.set_tick_params(labelsize=20)
        if sid == 1:
            axs[sid,1].set_xlabel(r'$y$', fontsize=ft)
            axs[sid,1].set_title(r'$x \leq s^{\rm CRPS}$', fontsize=ft)
        if sid == 0:
            axs[sid,1].set_title(r'$x \leq s^{\rm SSE}$', fontsize=ft)

        
        axs[sid,2].hist(data2, num_bins, facecolor=col, alpha=1, density=True, ec='white')
        axs[sid,2].plot(yaxvals, pdfexp, color='black')
        axs[sid,2].set_ylabel('Frequency/density', fontsize=ft)
        axs[sid,2].set_yticks([])
        axs[sid,2].xaxis.set_tick_params(labelsize=20)
        if sid == 1:
            axs[sid,2].set_xlabel(r'$y$', fontsize=ft)
            axs[sid,2].set_title(r'$x > s^{\rm CRPS}$', fontsize=ft)
        if sid == 0:
            axs[sid,2].set_title(r'$x > s^{\rm SSE}$', fontsize=ft)

    plt.savefig("Figure1.png", bbox_inches="tight")
    plt.close()