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

# Truncated lognormal (i.e. lognormal on [0,1])
size = 1000
dist = "lognormal"
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.05}
c_vals = [0.01,0.5,2]
tol_val = 0.001

all_results_lognormal_0_1 = []
for i in c_vals:
    
    revenue_params = {'a':3,'b':1,'c':i,'n':1,'m':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results_0_1 = rmm.model_sim(size,dist,dist_params,revenue_params,tol_val)
    all_results_lognormal_0_1.append(sim_results_0_1)


rmm.plot_sep_fun(all_results_lognormal_0_1,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','ineq_sep_plots','ln_plot_rho0_05.png'))


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


rmm.plot_sep_fun(all_results_lognormal_0_5,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','ineq_sep_plots','ln_plot_rho0_5.png'))



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


rmm.plot_sep_fun(all_results_lognormal_0_9,labels=["c=0.01","c=0.5","c=2"],output_path = os.path.join(os.getcwd(),'data', 'output','ineq_sep_plots','ln_plot_rho0_95.png'))