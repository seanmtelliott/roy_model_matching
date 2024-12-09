# Can we avoid phi hitting the boundary?
# Sort of.

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../../..')
print(os.getcwd())

# Import the "package"

sys.path.append("code/utilities/")
import roy_model_matching as rmm

# Set the parameters
size = 1000
dist = "lognormal"
dist_params = {'mean_k': 1,'mean_s': 0.5,'variance_k': 2,'variance_s':2,'correlation': 0}
c_vals = [0.01,0.5,1]

results = []
for i in c_vals:
    
    revenue_params = {'a':0.55,'b':0.45,'c':i,'n':1,'m':1,'cons':2}# where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_sep_fun(results,labels=["c=0.01","c=0.5","c=1"],output_path = os.path.join(os.getcwd(),'data', 'output','test_plots','a_equal_b.png'))


## When a approx equal b we don't get the boundary condition stuff, in fact, phi is just linear
## Try this with a >> b, now we get different results

results = []
for i in c_vals:
    
    revenue_params = {'a':0.9,'b':0.1,'c':i,'n':1,'m':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)
    results.append(sim_results)

# Generate the plots
rmm.plot_sep_fun(results,labels=["c=0.01","c=0.5","c=1"],output_path = os.path.join(os.getcwd(),'data', 'output','test_plots','a_not_equal_b.png'))