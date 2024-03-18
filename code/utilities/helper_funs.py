###############################################################################
# Sean Elliott - March 2024
#
# These are generic "helper" functions which are called by sim_methods.py
# That is, they are a layer further removed from the functions which the user
# would ever interact with.
###############################################################################

import numpy as np, statistics, scipy, pandas as pd

###############################################################################

## REVENUE FUNCTION
# The revenue function used in the simulation
def revenue(k,s,config):
    
    a = config['revenue']['coefficients']['a']
    n = config['revenue']['exponents']['n']
        
    b = config['revenue']['coefficients']['b']
    m = config['revenue']['exponents']['m']
    
    c = config['revenue']['coefficients']['c']
    
    rev = a*k**n + b*s**m + c*k*s 
    
    return rev

###############################################################################

## F TRANSFORM
# Determine the wages based on the zero-profit condition.
def f_transform(wage,key_types,sec_types,config,role="s"):
    
    if(role == "s"):
        wage_sec = []
        for k in range(len(key_types)):
            wage_sec.append(max([revenue(s, sec_types[k], config)- wage[np.where(key_types == s)[0][0]]  for s in key_types]))
        return wage_sec    

    if(role == "k"):
        pass

###############################################################################

## WORKER CHOICE
# Determine the mass of workers of each type who choose which role.
def worker_choice(adj_factor,N_type,wage_key,wage_sec,types,perform_check,config):    
    
    key_dist = [0] * N_type['k'][0]
    sec_dist = [0] * N_type['s'][0]

    key_worker_count = 0
    sec_worker_count = 0

    wage_key_altered = [x + adj_factor for x in wage_key]
    wage_sec_altered = [x - adj_factor for x in wage_sec]

    # Workers choose the occupation which has a greater wage 
        
    workers = types['workers']
 
    # Put the worker df together
    wage_fun_key = pd.DataFrame(np.column_stack([types['key'],wage_key_altered]),columns=("k","wage_k"))
    wage_fun_sec = pd.DataFrame(np.column_stack([types['sec'],wage_sec_altered]),columns=("s","wage_s"))
    worker_df = pd.DataFrame(workers,columns=("k","s"))
    worker_wages = worker_df.merge(wage_fun_key,on="k").merge(wage_fun_sec,on="s")
    
    # Select their roles
    worker_wages['role_sel_key'] = np.where(worker_wages["wage_k"]>worker_wages["wage_s"],1,0)
    worker_wages['role_sel_sec'] = np.where(worker_wages["wage_k"]<worker_wages["wage_s"],1,0)
    
    key_worker_count = sum(worker_wages['role_sel_key'])
    sec_worker_count = sum(worker_wages['role_sel_sec'])
    # Check the balance
    if perform_check == 1:
        diff = key_worker_count - sec_worker_count
        return diff
    if perform_check == 0:
        key_dist = np.array(worker_wages.groupby(['k'])['role_sel_key'].agg('sum'))
        sec_dist = np.array(worker_wages.groupby(['s'])['role_sel_sec'].agg('sum'))
        return key_dist,sec_dist
      
###############################################################################

## BALANCE WORKER DISTRIBUTIONS
# We want to adjust the key role wage to ensure that H(R) = G(R) = 1/2
def mass_balance(wage_k,wage_s,types,N,config):

    # Find the adjustment factor
    adj_factor = scipy.optimize.root_scalar(worker_choice, args=(N,wage_k,wage_s,types,1,config), method='bisect', bracket=(-10**6,10**6),xtol=10**(-5)).root
    wage_key_adj = [x + adj_factor for x in wage_k]
    wage_sec_adj = [x - adj_factor for x in wage_s]
    
    # Adjust wages and get the marginal skill distributions
    key_count, sec_count = worker_choice(adj_factor,N,wage_k,wage_s,types,0,config)
    
    # Adjust the zero values to ensure convergence
    non_zero_key = list(filter(lambda x: x > 0, key_count))
    non_zero_sec = list(filter(lambda x: x > 0, sec_count))

    key_count = [x if x > 0 else min(non_zero_key) / 10 for x in key_count]
    sec_count = [x if x > 0 else min(non_zero_sec) / 10 for x in sec_count]

    # Normalize the values (the OT solver doesn't like integers)
    key_count_norm = [i/sum(key_count) for i in key_count]
    sec_count_norm = [i/sum(sec_count) for i in sec_count]

    return key_count_norm, sec_count_norm, wage_key_adj, wage_sec_adj
###############################################################################

## CONVERGENCE CHECK
# Check to see if the iterative procedure has converged.
def convergence_check(wage_old,wage_new,tolerance):
    
    wage_new_mean = statistics.mean(wage_new)
    wage_old_mean = statistics.mean(wage_old)
    wage_diff = abs(wage_new_mean-wage_old_mean)
    
    if(wage_diff <= tolerance):
        check = 1
    elif(wage_diff > tolerance):
        check=0

    return check, wage_diff
###############################################################################
