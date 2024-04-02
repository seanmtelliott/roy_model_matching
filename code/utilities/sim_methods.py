###############################################################################
# Sean Elliott - March 2024
#
# This script contains functions called by roy_model_matching.py 
# these are not "visible" to the user in the sense that they need not interact
# with them to produce simulation results.
###############################################################################

# Import libraries

import numpy as np, pandas as pd, sys, itertools, ot, yaml
sys.path.append('code/utilities')
import helper_funs as helpers

###############################################################################


## MODIFY THE CONFIG
# This fills the fields of the template config yaml file based on user specification
def modify_config(size,dist,dist_params,revenue_params,tol=0.01,adj_wage=True):
    # Read in blank file
    with open('code/config/config.yaml','r') as file:
        config = yaml.safe_load(file)
        
    # Update with user-specified parameters
    
    # Economy size
    config['size']['num_types'] = size
    
    # Skill distribution
    
    if dist == "grid":
        config['distribution']['distribution_type'] = 'grid'
        config['distribution']['parameters']['mean'] = 'None'
        config['distribution']['parameters']['variance'] = 'None'
        config['distribution']['parameters']['correlation'] = 'None'
    elif dist == "lognormal":
        config['distribution']['distribution_type'] = 'lognormal'
        config['distribution']['parameters']['mean'] = dist_params['mean']
        config['distribution']['parameters']['variance'] = dist_params['variance']
        config['distribution']['parameters']['correlation'] = dist_params['correlation']
        pass
    
    # Revenue function
    config['revenue']['coefficients']['a'] = revenue_params['a']
    config['revenue']['coefficients']['b'] = revenue_params['b']
    config['revenue']['coefficients']['c'] = revenue_params['c']
    config['revenue']['exponents']['n'] = revenue_params['n']
    config['revenue']['exponents']['m'] = revenue_params['m']
    
    # Tolerance and adj_wage 
    config['tolerance'] = tol
    
    # for now take these as fixed
    
    return config

###############################################################################


## OUTPUT SIM PARAMETERS
# Print to the console the parameters of the simulation.
def sim_params(config):
    
    print("Running simulation with", config['size']['num_types']*config['size']['num_types'], "workers")
    print("Skills are distributed as",config['distribution']['distribution_type'],"with mean",config['distribution']['parameters']['mean'],
          "variance",config['distribution']['parameters']['variance'],"and correlation",config['distribution']['parameters']['correlation'])
    print("The revenue function is: F(k,s)=",config['revenue']['coefficients']['a'],"k^",config['revenue']['exponents']['n'],"+",
         config['revenue']['coefficients']['b'],"s^",config['revenue']['exponents']['m'],"+",
         config['revenue']['coefficients']['c'],"ks",sep='')
    
    return

###############################################################################


## GENERATE WORKERS
# Generate a list of workers and corresponding types based on the distribution specified.
def gen_workers(config):
    
    # Get size info
    N_types = pd.DataFrame([[config['size']['num_types'],config['size']['num_types']]],columns=['k','s']) 
    distribution = config['distribution']['distribution_type']
    
    if(distribution == "grid"):
        worker_types = np.array(list(itertools.product(np.arange(N_types['k'][0])/(N_types['k'][0]-1), 
                                                                  np.arange(N_types['s'][0])/(N_types['s'][0]-1))))
        
        key_types = np.unique(worker_types[:,0])
        sec_types = np.unique(worker_types[:,1])
        
        types = {}
        types['workers'] = worker_types
        types['key'] = key_types
        types['sec'] = sec_types
    if(distribution == "lognormal"):
        LN_mean = config['distribution']['parameters']['mean'] 
        LN_var = config['distribution']['parameters']['variance']
        LN_corr = config['distribution']['parameters']['correlation']
        
        # Generate lognormal skills and then only sample those for which both (k,s) < 1
        
        mu = [LN_mean, LN_mean]
        cov = np.array([[LN_var, LN_corr], [LN_corr, LN_var]])
        mvn = np.random.multivariate_normal(mu, cov, size=1000000)
        mvln = np.exp(mvn)
        mvn_samples_trunc = mvln[(mvln < 1).all(axis=1)]
        
        # Need to bin them into discrete chunks (take the hundreths place -- could change later)
        
        mvn_samples_trunc = mvn_samples_trunc.round(decimals=3)
        mvn_samples_trunc = mvn_samples_trunc[(mvn_samples_trunc > 0).all(axis=1)] #taking (0,1)

        # Sort them
        mvn_samples_sorted = np.array(sorted(sorted(mvn_samples_trunc,key=lambda e:e[1]),key=lambda e:e[0]))
        
        types = {}
        types['workers'] = mvn_samples_sorted
        types['key'] = np.unique(mvn_samples_sorted[:,0])
        types['sec'] = np.unique(mvn_samples_sorted[:,1])

        # Need to ensure that the number of types is equal for both sets
        if len(types['key']) != len(types['sec']):
            unique_types = list(set(types['key']) ^ set(types['sec']))
            for i in unique_types:
                types['key'] = np.delete(types['key'], np.where(types['key'] == i))
                types['sec'] = np.delete(types['sec'], np.where(types['sec'] == i))
                types['workers'] = types['workers'][~(types['workers'] == i).any(1),:]

    return types

###############################################################################


## COST MATRIX
# Determine the cost matrix for the OT map based on the specified revenue function.
def cost_matrix(types,config):
    workers = np.unique(types['workers'], axis=0)
    
    k_types = np.unique(workers[:,0])
    s_types = np.unique(workers[:,1])
    
    skill_pairs = pd.DataFrame(data = np.array(list(itertools.product(k_types, s_types))), columns = ['k','s'])
    skill_pairs['cost'] = -helpers.revenue(skill_pairs['k'],skill_pairs['s'],config)
    cost_mat = np.array(skill_pairs['cost']).reshape(len(k_types),len(s_types))
    
    return cost_mat

###############################################################################


## SET INITIAL WAGES
# The wages need to be set in some way for the first iteration -- this could be done any way, really.
# The random flag is used to determine wages in the random assignment/matching case 
def set_init_wage(types,config,random=False):
    
    if random==False:
    
        key_types = np.unique(types['key'])
        sec_types = np.unique(types['sec'])
    
        wage_key = [helpers.revenue(key_types[n],key_types[n],config)/2 for n in range(len(key_types))]
        wage_sec = helpers.f_transform(wage_key,key_types,sec_types,config, role="s")
    
        wages = {}
        wages['key'] = wage_key
        wages['sec'] = wage_sec
        
    elif random==True:
        
        key_types = np.unique(types['key'])
        sec_types = np.unique(types['sec'])
        wage_key = key_types * config['revenue']['coefficients']['a']
        wage_sec = sec_types * config['revenue']['coefficients']['b']
        
        wages = {}
        wages['key'] = wage_key
        wages['sec'] = wage_sec
    
    return wages

###############################################################################

### ITERATIVE STEP 
# Perform the iterative step in the simulation (i.e, get wages from OT map, check for convergence)
def get_optimal_wages(wages,cost_mat,types,config):
    
    # Collect relevant variables and initialize loop variables
    adj_wage = config['adj_wage']
    wage_key = wages['key']
    wage_sec = wages['sec']
    N_types = pd.DataFrame([[config['size']['num_types'],config['size']['num_types']]],columns=['k','s']) 
    tol = float(config['tolerance'])
    convg_check = 0
    iterations = 0 
    
    while convg_check==0:
    
        iterations += 1
    
        #Store wages from previous iteration for comparison purposes
        wage_key_old = wage_key
        wage_sec_old = wage_sec
    
        ## Step 1: Induce a half cut of the workers and collect the marginal skill distributions
    
        key_dist, sec_dist, wage_key_adj, wage_sec_adj = helpers.mass_balance(wage_key_old.copy(),wage_sec_old.copy(),types,N_types,config)
    
        ## Step 2: OT problem

        ot_results = ot.lp.emd(key_dist,sec_dist, cost_mat, log=True)
        wage_key = ([-x for x in ot_results[1]['u']])
        wage_sec = ([-x for x in ot_results[1]['v']])

        ## Step 3: Check for convergence in key-wage function

        convg_check, abs_diff = helpers.convergence_check(wage_key_old,wage_key,tol)

        # Step 4: Return the wage function and OT results if convergence attained (also stop at 100 iterations)
        if convg_check==1:
            print("Convergence in wage function attained after", iterations,"iterations.")
            if adj_wage == True:
                results = {}
                results['wage_key'] = wage_key_adj
                results['wage_sec'] = wage_sec_adj
                results['matching_fun'] = [types['sec'][ot_results[0][k].argmax()] for k in range(len(types['key']))]
                results['key_dist'] = key_dist
                results['sec_dist'] = sec_dist
                results['ot_mat'] = ot_results[0]
            elif adj_wage == False:
                results = {}
                results['wage_key'] = wage_key
                results['wage_sec'] = wage_sec
                results['matching_fun'] = [types['sec'][ot_results[0][k].argmax()] for k in range(len(types['key']))]
                results['key_dist'] = key_dist
                results['sec_dist'] = sec_dist
                results['ot_mat'] = ot_results[0]
            break
        elif convg_check == 0:
            print("Iteration",iterations,": Not converged, difference is", abs_diff)
    
        
    return results

###############################################################################


## GET FIRM INFO
# Use wages and matching function to compile firm info into a dataframe
def get_firm_info(results,types,config):
    
    wages = pd.DataFrame(np.column_stack((types['key'],types['sec'],results['wage_key'],results['wage_sec'],
                             np.log(results['wage_key']),np.log(results['wage_sec']))),columns=['k','s','wage_key','wage_sec','log_wage_key','log_wage_sec'])
    
    matching_fun = pd.DataFrame(np.column_stack((types['key'],results['matching_fun'])),
                                columns=['k','s'])
   
    match_wages_s = matching_fun.merge(wages[['s','wage_sec']], on = 's', how = 'left')
    
    firms = pd.DataFrame(np.column_stack((matching_fun,results['wage_key'],match_wages_s['wage_sec'],
                                          np.log(results['wage_key']),np.log(match_wages_s['wage_sec']),np.log(results['wage_key'])-np.log(match_wages_s['wage_sec']))),
                                         columns=['k','s','wage_key','wage_sec','log_wage_key','log_wage_sec','diff'])
    

    
    return firms

## GET POP WEIGHTS
# Need to combine the firm info with info about number of each type to determine weights in the pop
def get_pop_weights(sim_results):

    ot_mat = np.floor(sim_results['ot']['ot_mat'] * len(sim_results['types']['workers']))
    firms = sim_results['firms']
    
    sec_wage = pd.DataFrame(np.column_stack((sim_results['types']['sec'],
                                             sim_results['ot']['wage_sec'],np.log(sim_results['ot']['wage_sec']))),
                            columns=["s","wage_sec","log_wage_sec"])
    sec_wage.drop_duplicates(subset=['s'], keep="first",inplace=True)
    
    match_count = []
    for i in range(len(ot_mat[0])):
        key_type = sim_results['types']['key'][i]
        for j in range(len(ot_mat[0])):
            sec_type = sim_results['types']['key'][j]
            count = ot_mat[i][j]
            match_count.append([key_type,sec_type,count])

    match_count = np.array(match_count)
    rows=np.where(match_count[:,2]!=0)
    match_count = pd.DataFrame(match_count[rows],columns=['k','s','times'])
    
    match_count = match_count.merge(firms[['k','wage_key','log_wage_key']], on = 'k',how='left')
    match_count = match_count.merge(sec_wage,on = "s",how = 'left')
    expanded_matches = match_count.loc[match_count.index.repeat(match_count.times)].reset_index(drop=True)
    
    output = []
    for i in range(len(expanded_matches)):
        output.append(helpers.revenue(expanded_matches['k'][i], expanded_matches['s'][i], sim_results['config']))
    
    expanded_matches['firm_avg'] = (expanded_matches['log_wage_key'] + expanded_matches['log_wage_sec'])/2
    expanded_matches['firm_output'] = np.log(output)
    expanded_matches['resid_key'] = expanded_matches['log_wage_key'] - expanded_matches['firm_avg']
    expanded_matches['resid_sec'] = expanded_matches['log_wage_sec'] - expanded_matches['firm_avg']
    
    return expanded_matches

## COMPUTE AVG WITHIN PERCENTILES
# This is how it is done in the Bloom paper -- trying to replicate that exactly.
def get_percentile_avg(firms,series):
    
    if series == "individual":
        key_wage = firms['log_wage_key']
        sec_wage = firms['log_wage_sec']
        all_wages = pd.DataFrame(pd.concat([key_wage,sec_wage]),columns=["wages"])
        all_wages['perc_rank'] = all_wages['wages'].rank(pct=True).round(2)
        percentile_avg_ind = all_wages.groupby(['perc_rank']).mean()
        return percentile_avg_ind
    
    if series == "firm":
        key_wage = firms['log_wage_key']
        sec_wage = firms['log_wage_sec']
        all_wages = pd.DataFrame((key_wage + sec_wage)/2,columns=["avg_wage"])
        all_wages['perc_rank'] = all_wages['avg_wage'].rank(pct=True).round(2)
        percentile_avg_firm = all_wages.groupby(['perc_rank']).mean()
        return percentile_avg_firm
    
    if series == "within":
        key_wage = firms['log_wage_key']
        sec_wage = firms['log_wage_sec']
        avg_wage = (key_wage + sec_wage)/2
        key_wage_resid = key_wage - avg_wage
        sec_wage_resid = sec_wage - avg_wage
        all_wages = pd.DataFrame(pd.concat([key_wage_resid,sec_wage_resid]),columns=["wage_resid"])
        all_wages['perc_rank'] = all_wages['wage_resid'].rank(pct=True).round(2)
        percentile_avg_within = all_wages.groupby(['perc_rank']).mean()
        return percentile_avg_within

    return 