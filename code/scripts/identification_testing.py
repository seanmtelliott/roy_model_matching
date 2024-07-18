# This script is to just play around with the simulations -- testing purposes

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../..')


# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

# Set the parameters
size = 1000
dist = "lognormal"
dist_params = {'mean': 0,'variance': 1,'correlation': 0.5}
revenue_params = {'a':0.75,'b':0.25,'c':1,'n':1,'m':1,'cons':2} # where F(k,s) = a*k**n + b*s**m + c*k*s
rho_vals = [0.25,0.5,0.75]

results = []
for i in rho_vals:
    
    dist_params = {'mean': 0,'variance': 1,'correlation': i}
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_indentification(results,labels=["rho=0.25","rho=0.5","rho=0.75"],output_path = os.path.join(os.getcwd(),'data', 'indentification_test','rho_varying.png'))

