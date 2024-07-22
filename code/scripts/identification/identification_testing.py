# Can we avoid phi hitting the boundary?
# Sort of.

# Import libraries

import sys, os

# Set working directory to be root directory of the repository

os.chdir(os.path.dirname(__file__))
os.chdir('../../..')


# Import the "package"

sys.path.append("code/utilities")
import roy_model_matching as rmm

# Set the parameters
size = 1000
dist = "lognormal"
dist_params = {'mean': 0,'variance': 1,'correlation': 0.5}
c_vals = [0.01,0.5,1]

results = []
for i in c_vals:
    
    revenue_params = {'a':0.55,'b':0.45,'c':i,'n':1,'m':1, 'cons':1} # where F(k,s) = a*k**n + b*s**m + c*k*s
    
    sim_results = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.005)
    results.append(sim_results)

# Generate the plots
rmm.plot_sep_fun(results,labels=["c=0.01","c=0.5","c=1"],output_path = os.path.join(os.getcwd(),'data', 'output','identification','a_equal_b.png'))
