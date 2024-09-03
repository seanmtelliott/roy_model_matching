# Here I will use results from HH to see if we can recover the underlying skill distribution in the linear separation case.
#
# Import libraries

import os, numpy as np, pandas as pd, statistics as stats, scipy
from scipy.stats import norm
from scipy.optimize import fsolve, least_squares
# Set working directory to be root directory of the repository

# os.chdir(os.path.dirname(__file__))
# os.chdir('../..')


# Generate some workers

mu = [1/2, 1/2]
cov = np.array([[2, 0], [0, 2]])
mvn = np.random.multivariate_normal(mu, cov, size=1000000)
workers = pd.DataFrame(mvn,columns=['k','s'])

a=1
b=1

# Occupational choice
workers['key_select'] = np.where(workers["k"]>workers["s"],1,0)

# Get moments of conditional distribution

pr_key_sel = sum(workers['key_select'])/len(workers)

key_df = workers[workers['key_select']==1]
sec_df = workers[workers['key_select']==0]

mean_key = stats.mean(key_df['k'])
mean_sec = stats.mean(sec_df['s'])

var_key = stats.variance(key_df['k'])
var_sec = stats.variance(sec_df['s'])

skew_key = scipy.stats.skew(key_df['k'])
skew_sec = scipy.stats.skew(sec_df['s'])

# Define inverse mills ratio function
def inv_mills(x):
    return norm.pdf(x)/norm.cdf(x)

sqrt = np.emath.sqrt

# Solve a system of nonlinear equations
# x[0] : mu_k
# x[1] : mu_s
# x[2] : sigma2_k
# x[3] : sigma2_s
# x[4] : sigma_ks
def func(x):
    D = np.abs((x[0]-x[1])/sqrt(x[2]**2+x[3]**2-2*x[4]))
    tau_k = np.abs((x[2]**2-x[4])/sqrt(x[2]**2+x[3]**2-2*x[4]))
    tau_s = np.abs((x[3]**2-x[4])/sqrt(x[2]**2+x[3]**2-2*x[4]))
    return [pr_key_sel - norm.cdf(D),
            mean_key - x[0] - tau_k * inv_mills(D),
            mean_sec - x[1] - tau_s * inv_mills(-D),
            var_key - x[2] - (tau_k**2) * (-inv_mills(D)*D - (inv_mills(D))**2),
            var_sec - x[3] - (tau_s**2) * (inv_mills(-D)*D - (inv_mills(-D))**2)]

root = fsolve(func, [1/2,1/2,2,2,0])
print(root)

