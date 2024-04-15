
## Sean Elliott - April 2024
# 
# Want to simulate an economy which coincides with the experience observed in Brazil from 1999-2013
# What we should get in this scenario is that within-firm inequality is roughly constant and
# that inequality across firms and across workers goes down as we move up the distribution
# that is, in contrast to the US experience, the line is downward sloping.

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

## First set of parameters
dist_params = {'mean': 0,'variance': 1,'correlation': 0.05}
revenue_params = {'a':0.75,'b':0.25,'c':1.75,'n':1,'m':1,'cons':2}
scenario1 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

## Second set of parameters
dist_params = {'mean': 0.5,'variance': 1,'correlation': 0.9}
revenue_params = {'a':2.5,'b':0.25,'c':0.5,'n':2,'m':2,'cons':4}
scenario2 = rmm.model_sim(size,dist,dist_params,revenue_params,tolerance=0.001)

inequality_sims = {}
inequality_sims['scenario1'] = scenario1
inequality_sims['scenario2'] = scenario2

rmm.plot_inequality(inequality_sims,labels=["scenario1","scenario2"],
                    output_path = os.path.join(os.getcwd(),'data', 'output','brazil_sim','test.png'))