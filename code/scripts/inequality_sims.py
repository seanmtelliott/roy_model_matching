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
os.chdir('../..')

# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

### Set rho = 0.5

# Set the parameters
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.25}

## MODEL w/ linear phi
#Run the model with c=0 or c very close to 0
revenue_params = {'a':3,'b':1,'c':0.01,'n':1,'m':1}
linear_sep_50 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

# MODEL w/ non-linear phi
#Set c s.t. we get non-linear phi (here I am choosing c=2, same as previous simulations)

revenue_params = {'a':3,'b':1,'c':2,'n':1,'m':1}
nonlinear_sep_50 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

inequality_sims_50 = {}
inequality_sims_50['linear'] = linear_sep_50
inequality_sims_50['nonlinear'] = nonlinear_sep_50

rmm.plot_inequality(inequality_sims_50,labels=["Linear","Non-linear"],
                    output_path = os.path.join(os.getcwd(),'data', 'output','ineq_sep_plots','inequality_rho_test.png'))

### Set rho = 0.05 -- these don't change much so probably can only just show the rho=0.5 

# # Set the parameters
# dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.05}

# ## MODEL w/ linear phi
# #Run the model with c=0 or c very close to 0
# revenue_params = {'a':3,'b':1,'c':0.01,'n':1,'m':1}
# linear_sep_05 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

# # MODEL w/ non-linear phi
# #Set c s.t. we get non-linear phi (here I am choosing c=2, same as previous simulations)

# revenue_params = {'a':3,'b':1,'c':2,'n':1,'m':1}
# nonlinear_sep_05 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

# inequality_sims_05 = {}
# inequality_sims_05['linear'] = linear_sep_05
# inequality_sims_05['nonlinear'] = nonlinear_sep_05

# rmm.plot_inequality(inequality_sims_05,labels=["Random","Linear","Non-linear"],
#                     output_path = os.path.join(os.getcwd(),'data', 'output','ineq_sep_plots','inequality_rho_0_05.png'))

# ### Set rho = 0.95

# # Set the parameters
# dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.95}

# ## MODEL w/ linear phi
# #Run the model with c=0 or c very close to 0
# revenue_params = {'a':3,'b':1,'c':0.01,'n':1,'m':1}
# linear_sep_95 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

# # MODEL w/ non-linear phi
# #Set c s.t. we get non-linear phi (here I am choosing c=2, same as previous simulations)

# revenue_params = {'a':3,'b':1,'c':2,'n':1,'m':1}
# nonlinear_sep_95 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

# inequality_sims_95 = {}
# inequality_sims_95['linear'] = linear_sep_95
# inequality_sims_95['nonlinear'] = nonlinear_sep_95

# rmm.plot_inequality(inequality_sims_95,labels=["Random","Linear","Non-linear"],
#                     output_path = os.path.join(os.getcwd(),'data', 'output','ineq_sep_plots','inequality_rho_0_95.png'))