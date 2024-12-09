## Sean Elliott - March 2024
# 
# Here we want to see how inequality changes when we incorporate different features of the model.
# We can see what happens as we allow workers to interact with one another in production
# That is, what happens when we introduce the non-linearity in the separating function?
# The goal is to show that this roughly produces similar results to Bloom et al. (2019)

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../../..')

# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

### Set rho = 0.5

# Set the parameters
size = 1000
dist = "lognormal"
dist_params = {'mean_k': 0, 'mean_s': 0,'variance_k': 1,'variance_s': 1,'correlation': 0}

## First set of parameters
revenue_params = {'a':0.55,'b':0.45,'c':0.01,'n':1,'m':1,'cons':2}
scenario1 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

## Second set of parameters
dist_params = {'mean_k': 0, 'mean_s': 0,'variance_k': 1,'variance_s': 1,'correlation': 0.5}
revenue_params = {'a':1.6,'b':0.4,'c':0.15,'n':2,'m':1,'cons':2}
scenario2 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

inequality_sims = {}
inequality_sims['scenario1'] = scenario1
inequality_sims['scenario2'] = scenario2

rmm.plot_inequality(inequality_sims,labels=["scenario1","scenario2"],
                    output_path = os.path.join(os.getcwd(),'data', 'output','test_plots','inequality_1983_2013_match.png'))

## Plot cross-sectional inequality

rmm.plot_ineq_cross_sect(scenario2,output_path = os.path.join(os.getcwd(),'data', 'output','US_ineq_cross_sect.png'))
