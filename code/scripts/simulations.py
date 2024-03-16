# Sean Elliott - March 2024
#
# Simulations for "A generalization of a model of Roy for partition of labour force via matching and occupational choice"
#
# This script performs the simulation with various configs and outputs plots
# The user can choose which simulation results to overlay on the same plot


## Rectangular skill distribution

# Import libraries

import sys, os

# Set working directory

os.chdir("/home/selliott/Research/roy_model_matching")

# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

# Rectangular grid distribution with different values of 'c' in the revenue function
size = 1000
dist = "grid"
dist_params = None
c_vals = [0.01,1.5,3]

all_results_grid = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':2,'m':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_grid = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_grid.append(sim_results_grid)

# Generate the plots
rmm.gen_plots(all_results_grid,labels=["c=0.01","c=1.5","c=3"],output_path = os.path.join(os.getcwd(),'data', 'output','grid_plot.png'))
# Write the results to disk
#rmm.save_results(all_results_grid,labels=["c=0.01","c=1.5","c=3"],output_path = os.path.join(os.getcwd(),'data', 'output','worker_summary.csv'))

## Lognormal with different values of c and rho

# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.5}
c_vals = [0.01,1.5,3]

all_results_lognormal_0_50 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':2,'m':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_50 = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_lognormal_0_50.append(sim_results_0_50)


rmm.gen_plots(all_results_lognormal_0_50,labels=["c=0.01","c=1.5","c=3"],output_path = os.path.join(os.getcwd(),'data', 'output','ln_plot_rho0_5.png'))

# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.1}
c_vals = [0.01,1.5,3]

all_results_lognormal_0_10 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':2,'m':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_10 = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_lognormal_0_10.append(sim_results_0_10)


rmm.gen_plots(all_results_lognormal_0_10,labels=["c=0.01","c=1.5","c=3"],output_path = os.path.join(os.getcwd(),'data', 'output','ln_plot_rho0_10.png'))


# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.7}
c_vals = [0.01,1.5,3]

all_results_lognormal_0_70 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':2,'m':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_70 = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_lognormal_0_70.append(sim_results_0_70)


rmm.gen_plots(all_results_lognormal_0_70,labels=["c=0.01","c=1.5","c=3"],output_path = os.path.join(os.getcwd(),'data', 'output','ln_plot_rho0_70.png'))

# Fix value of c and change rho

size = 1000
dist = "lognormal"
revenue_params = {'a':3,'b':1,'c':1.5,'n':2,'m':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
rho_vals = [0.01,0.1,0.5,0.9,0.99]

all_results_lognormal_rho = []
for i in rho_vals:
    
    dist_params = {'mean': 0.5,'variance': 1,'correlation': i}
    
    sim_results_rho = rmm.model_sim(size,dist,dist_params,revenue_params)
    all_results_lognormal_rho.append(sim_results_rho)


rmm.gen_plots(all_results_lognormal_rho,labels=["rho=0.01","rho=0.1","rho=0.5","rho=0.9","rho=0.99"],output_path = os.path.join(os.getcwd(),'data', 'output','ln_plot_rho_varying.png'))
