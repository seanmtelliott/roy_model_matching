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
revenue_params = {'a':1,'b':1,'c':1,'n':1,'m':1,'cons':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
dist_params = {'mean_k': 0,'mean_s': 0,'variance_k': 1,'variance_s':1,'correlation': 0}
sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
plt.plot(sim_results['ot']['matching_fun'])