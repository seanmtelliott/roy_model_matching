# Sean Elliott - March 2024
#
# Showing different results for linearity of separating function



## Rectangular skill distribution

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../..')

# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

# Rectangular grid distribution with different values of 'c' in the revenue function
size = 1000
dist = "grid"
dist_params = None
c_vals = [0.01,0.5,2]

all_results_grid = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':2,'m':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_grid = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_grid.append(sim_results_grid)

# Generate the plots
rmm.gen_plots(all_results_grid,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','sep_function_test','grid_plot_paper.png'))

## This almost becomes more linear as c gets larger. What is going on here?
## Lets remove the exponents on k and s (this is more like the cobb-douglas form we'd see typically)

# Rectangular grid distribution with different values of 'c' in the revenue function
size = 1000
dist = "grid"
dist_params = None
c_vals = [0.01,0.5,2]

all_results_grid = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':1,'m':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_grid = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_grid.append(sim_results_grid)

# Generate the plots
rmm.gen_plots(all_results_grid,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','sep_function_test','grid_plot_linear_ks.png'))

## Now we get linearity when c -> 0
## Using a similar distribution from HH (lognormal, sigma11 > sigma12)
## Start with low correlation (rho=0.25)

# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.05}
c_vals = [0.01,0.5,2]


all_results_lognormal_0_1 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':1,'m':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_1 = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_lognormal_0_1.append(sim_results_0_1)


rmm.gen_plots(all_results_lognormal_0_1,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','sep_function_test','ln_plot_rho0_05.png'))


# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.5}
c_vals = [0.01,0.5,2]
tol_val = 0.001

all_results_lognormal_0_5 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':1,'m':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_5 = rmm.model_sim(size,dist,dist_params,revenue_params,tol_val)
    all_results_lognormal_0_5.append(sim_results_0_5)


rmm.gen_plots(all_results_lognormal_0_5,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','sep_function_test','ln_plot_rho0_5.png'))



## Go to a higher correlation (rho=0.75)

# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.95}
c_vals = [0.01,0.5,2]

all_results_lognormal_0_9 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':1,'m':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_9 = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_lognormal_0_9.append(sim_results_0_9)


rmm.gen_plots(all_results_lognormal_0_9,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','sep_function_test','ln_plot_rho0_95.png'))