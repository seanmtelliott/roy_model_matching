# This script is to just play around with the simulations -- testing purposes

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../../../')


# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

# Set the fixed parameters
size = 1000
dist = "lognormal"
revenue_params = {'a':0.75,'b':0.25,'c':1,'n':1,'m':1,'cons':2} # where F(k,s) = a*k**n + b*s**m + c*k*s


## Symmetric cases: Change mu, sigma, rho but equally for both sectors

# Modifying the correlation between skills

rho_vals = [0.25,0.5,0.75]
results = []
for i in rho_vals:
    
    dist_params = {'mean_k': -1,'mean_s':-1,'variance_k': 1,'variance_s':1,'correlation': i}
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_indentification(results,labels=["rho=0.25","rho=0.5","rho=0.75"],output_path = os.path.join(os.getcwd(),'data','output', 'identification_test','symmetric'),file_name=["rho_varying.png","rho_varying_obs.png"])


# Modifying the average skill level

mu_vals = [-3,-2,-1]
results = []
for i in mu_vals:
    
    dist_params = {'mean_k': i,'mean_s':i,'variance_k': 1,'variance_s':1,'correlation': 0.5}
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_indentification(results,labels=["mu=-3","mu=-2","mu=-1"],output_path = os.path.join(os.getcwd(),'data','output', 'identification_test','symmetric'),file_name=["mu_varying.png","mu_varying_obs.png"])

# Changing the variance in skill level

sigma_vals = [1,2,3]
results = []
for i in sigma_vals:
    
    dist_params = {'mean_k': -1,'mean_s':-1,'variance_k': i,'variance_s':i,'correlation': 0.5}
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_indentification(results,labels=["sigma=1","sigma=2","sigma=3"],output_path = os.path.join(os.getcwd(),'data','output', 'identification_test','symmetric'),file_name=["sigma_varying.png","sigma_varying_obs.png"])


## Asymmetric cases: Change mu and sigma separately for sectors

# Mu varying
mu_vals = [-1.5,-1,-0.5]
results = []
for i in mu_vals:
    
    dist_params = {'mean_k': i,'mean_s':-1,'variance_k': 1,'variance_s':1,'correlation': 0.5}
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_indentification(results,labels=["mu=-1.5","mu=-1","mu=-0.5"],output_path = os.path.join(os.getcwd(),'data','output', 'identification_test','asymmetric'),file_name=["mu_k_varying.png","mu_k_varying_obs.png"])

# Sigma varying

sigma_vals = [1,2,3]
results = []
for i in sigma_vals:
    
    dist_params = {'mean_k': -1,'mean_s':-1,'variance_k': i,'variance_s':1,'correlation': 0.5}
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_indentification(results,labels=["sigma=1","sigma=2","sigma=3"],output_path = os.path.join(os.getcwd(),'data','output', 'identification_test','asymmetric'),file_name=["sigma_k_varying.png","sigma_k_varying_obs.png"])
