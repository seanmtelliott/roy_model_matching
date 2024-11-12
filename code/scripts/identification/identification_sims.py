# This script is to just play around with the simulations -- testing purposes

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../../../')


# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm
import statistics as stats, pandas as pd, matplotlib.pyplot as plt

# Set the fixed parameters
size = 1000
dist = "lognormal"

means = []
for i in range(25):
    revenue_params = {'a':1,'b':1,'c':0.5,'n':1,'m':1,'cons':0} # where F(k,s) = a*k**n + b*s**m + c*k*s
    dist_params = {'mean_k': 0.5,'mean_s': 0,'variance_k': 1,'variance_s':1,'correlation': 0}
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.01)
    est_wages = rmm.get_estimated_moments(sim_results)
    means.append(est_wages)

means_df = pd.DataFrame(means,columns=["k","s"]) - 5
mean_k = stats.mean(means_df['k'])
mean_s = stats.mean(means_df['s'])


# Distribution of the means
plt.hist(means_df['k'],bins=1, color='red', edgecolor='black')
plt.axvline(mean_s, color='k', linestyle='dashed', linewidth=1)

plt.hist(means_df['s'],bins=1, color='skyblue', edgecolor='black')
plt.axvline(mean_k, color='k', linestyle='dashed', linewidth=1)
