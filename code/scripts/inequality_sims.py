## Sean Elliott - March 2024
# 
# This script will produce simulation results to compare levels of inequality across different economies.
# First, we start with a completely random economy. Workers are assigned skills, and then they
# are placed into jobs at random, where wages are simply determined by the zero profit condition
#
# Second, we simulate a roy economy where workers choose their roles based on prevailing wages
# the goal here is to show that worker sorting will increase inequality
#
# Finally, we add the full model which includes matching and the non-linear separation function
# we want to see what happens to wage inequality once we incorporate the full specification
# 
# Skills will be lognormal and we use the revenue function F(k,s) = ak + bs + cks
# Wages are given by pi(k) = p_k * k and w(s) = p_s * s in the first two cases
# and are optimally determined by the OT problem in the 3rd case


# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../..')

# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

# First simulate the full model

size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.5}
revenue_params = {'a':3,'b':1,'c':1,'n':1,'m':1}

full_model_sim = rmm.model_sim(size,dist,dist_params,revenue_params)

# Match them randomly

random_matching = rmm.randomize_results(full_model_sim,randomized = "matching")

rmm.plot_inequality(full_model_sim,random_matching)

# Assign them to roles randomly and match them randomly

#random_economy = rmm.randomize_results(full_model_sim,randomized = "all")