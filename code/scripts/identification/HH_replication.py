# Here I will use results from HH to see if we can recover the underlying skill distribution in the linear separation case.
#
# Import libraries

import os, numpy as np, pandas as pd, statistics as stats, scipy
from scipy.stats import norm
from scipy.optimize import fsolve, least_squares
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt

# Set working directory to be root directory of the repository

# os.chdir(os.path.dirname(__file__))
# os.chdir('../..')


# Define inverse mills ratio function
def inv_mills(x):
    return norm.pdf(x)/norm.cdf(x)

# Generate some workers
mean_k = []
mean_s = []
for i in range(999):
    mu = [3/4, 1/2]
    cov = np.array([[2, 0], [0, 2]])
    mvn = np.random.multivariate_normal(mu, cov, size=10000)
    workers = pd.DataFrame(mvn,columns=['k','s'])

    a=1
    b=1

    # Wages
    workers['key_wage'] = a*workers['k']
    workers['sec_wage'] = b*workers['s']

    # Occupational choice
    workers['key_select'] = np.where(workers['key_wage']>workers['sec_wage'] ,1,0)

    # Get moments of conditional distribution

    pr_key_sel = sum(workers['key_select'])/len(workers)

    key_df = workers[workers['key_select']==1]
    sec_df = workers[workers['key_select']==0]

    mean_key = stats.mean(key_df['key_wage'])
    mean_sec = stats.mean(sec_df['sec_wage'])

    var_key = stats.variance(key_df['key_wage'])
    var_sec = stats.variance(sec_df['sec_wage'])

    key_df["mean_dev3"] = (key_df['key_wage']-mean_key)**3
    sec_df["mean_dev3"] = (sec_df['sec_wage']-mean_sec)**3

    skew_key = stats.mean(key_df["mean_dev3"])
    skew_sec =  stats.mean(sec_df["mean_dev3"])

    D = norm.ppf(pr_key_sel)
    lD = inv_mills(D)
    lDneg = inv_mills(-D)

    tau_k = (skew_key/(lD * (2*lD**2 + 3 * D * lD + D**2 - 1)))**(1/3)
    tau_s = (skew_sec/(lDneg * (2*lDneg**2 - 3 * D * lDneg + D**2 - 1)))**(1/3)
    mu_k = mean_key - tau_k * lD
    mu_s = mean_sec - tau_s * lDneg
    sigma2_k = var_key - tau_k**2 * ((-lD)*D - lD**2)
    sigma2_s = var_sec - tau_s**2 * ((lDneg)*D - lDneg**2)
    sigma_ks = (-(mu_k**2)+2*mu_k*mu_s - mu_s**2 + sigma2_k*(D**2) + sigma2_s*(D**2))/(2*D**2)

    mean_k.append(mu_k)
    mean_s.append(mu_s)
    
    
# Distribution of the means
plt.hist(mean_k,bins=30, color='red', edgecolor='black')
plt.axvline(stats.mean(mean_k), color='k', linestyle='dashed', linewidth=1)

plt.hist(mean_s,bins=30, color='skyblue', edgecolor='black')
plt.axvline(stats.mean(mean_s), color='k', linestyle='dashed', linewidth=1)
