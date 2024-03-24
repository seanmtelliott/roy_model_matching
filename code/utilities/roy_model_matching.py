################################################################################
# Sean Elliott - March 2024
#
# Simulations for "A generalization of a model of Roy for partition of labour force via matching and occupational choice"
#
# These are the callable functions from the perspective of the user.
# It contains the main functionality:
# 1. Perform simulation
# 2. Generate plots based on simulation results
#TODO: 3. Output the results in a .csv file
################################################################################

# Import libraries
import sys, matplotlib.pyplot as plt, logging, numpy as np, copy, statistics
logging.getLogger().setLevel(logging.CRITICAL)
sys.path.append('code/utilities')
import sim_methods as sm, helper_funs as helpers

################################################################################

## PERFORM SIMULATION

def model_sim(size,dist,dist_params,revenue_params,tolerance=0.01):
    
    # Set the config file
    
    config = sm.modify_config(size, dist, dist_params, revenue_params,tolerance)
    
    #Display to the user the simulation being run
    sm.sim_params(config)
    
    # Get type information
    types = sm.gen_workers(config)

    # Cost matrix
    cost_mat = sm.cost_matrix(types.copy(),config)

    # Set the initial key wages to be some increasing function of type
    wages = sm.set_init_wage(types.copy(),config)

    # Get the results from the OT problem (solves iteratively)

    ot_results = sm.get_optimal_wages(wages,cost_mat,types,config)
    
    # Organize firm information (useful for inequality plots later)
    
    firms = sm.get_firm_info(ot_results,types,config)
    
    # Collect all of the relevant information and store it as a dict
    results = {}
    results['ot'] = ot_results
    results['types'] = types
    results['config'] = config
    results['firms'] = firms
    
    return results

###############################################################################

## GENERATE SEPARATING FUNCTION PLOTS

def plot_sep_fun(results,labels,output_path):
    
    # Put the plots together
    fig, axes = plt.subplots(2,2)
    
    # TODO: generalize this for different number of labels (only handles 3 and 5 at the moment)
    # Specify some colours
    if len(labels) == 3:
        colors = ['grey','brown','k']
    if len(labels) == 5:
        colors = ['grey','brown','k','blue','green']
        
    for i in range(len(labels)):
        
        # Get number of ticks
        types_key = results[i]['types']['key']
        types_sec = results[i]['types']['sec']
        num_types = len(types_key)
    
        # Get wages
        wage_key = np.log(results[i]['ot']['wage_key'])
        wage_sec = np.log(results[i]['ot']['wage_sec'])
    
        # Wages
        if i < len(labels)-1:
            wage_lab = ["_Hidden","_Hidden"]
        elif i==len(labels)-1:
            wage_lab = ["k","s"]
            
        axes[0,0].plot(types_key,wage_key,color=colors[i],label=wage_lab[0])
        axes[0,0].plot(types_sec,wage_sec,linestyle="dotted",color=colors[i],label=wage_lab[1])
        axes[0,0].legend(loc='upper left')
        axes[0,0].set_title("Wages by type")
        axes[0,0].set_xlabel('Skill level')
        axes[0,0].set_ylabel('Wage')

        # Separating function
        wage_differential = [[wage_key[k]-wage_sec[s] for s in range(num_types)] for k in range(num_types)]
        wage_differential_abs = [[abs(wage_key[k]-wage_sec[s]) for s in range(num_types)] for k in range(num_types)]
        sep_function = [wage_differential_abs[k].index(min(wage_differential_abs[k]))/(num_types-1) for k in range(num_types)]
        axes[0,1].plot(types_key,sep_function,color=colors[i],label = labels[i])
        axes[0,1].set_title("Separating function")
        axes[0,1].set_xlabel('k')
        axes[0,1].set_ylabel('s')
    
        # Matching function
        matching_fun = [types_sec[results[i]['ot']['ot_mat'][k].argmax()] for k in range(num_types)] 
        axes[1,0].plot(types_key,matching_fun,color=colors[i],label = labels[i])
        axes[1,0].set_title("Matching function")
        axes[1,0].set_xlabel('k')
        axes[1,0].set_ylabel('s')
    
        # Wage inequality
        ineq_function = [wage_differential[k][int(matching_fun[k]*(num_types-1))] for k in range(num_types)]
        axes[1,1].plot(types_key,ineq_function,color=colors[i],label = labels[i])
        axes[1,1].set_title("Wage inequality in matches")
        axes[1,1].set_xlabel('k')
        axes[1,1].set_ylabel('Wage difference')
        
    lines_labels = [fig.axes[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Position the legend differently depending on how many labels we have 
    # TODO: generalize this also
    if len(labels) == 3:
        fig.legend(lines, labels, loc="lower center",ncol=(len(labels)),  prop = { "size": 7.5 })
    if len(labels) == 5:
        fig.legend(lines, labels, bbox_to_anchor=(0.44, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })

    
    plt.tight_layout()
    plt.savefig(output_path)

    
    return 

###############################################################################

## GENERATE INEQUALITY RESULTS
def plot_inequality(result1,result2):
    
    # First look at inequality across firms -- do we see more inequality for more productive firms?
    firms = result1['firms']
    firms_random = result2['firms']
    
    firm_output = []
    firm_output_random = []
    for i in range(len(firms)):
        
        firm_output.append(helpers.revenue(firms['k'][i],firms['s'][i],result1['config']))
        firm_output_random.append(helpers.revenue(firms_random['k'][i],firms_random['s'][i],result2['config']))

    percentile_orig = statistics.quantiles(firm_output,n=100)    
    percentile_rand = statistics.quantiles(firm_output_random,n=100)    
    diff = np.subtract(percentile_orig, percentile_rand)
    plt.plot(diff)
    
    within_ineq = statistics.quantiles(firms['diff'],n=100)
    within_ineq_random = statistics.quantiles(firms_random['diff'],n=100)
    diff_ineq = np.subtract(within_ineq, within_ineq_random)
    plt.plot(diff_ineq)

    # Next look at inequality within

    return

## RANDOMIZE RESULTS
# Take results from the full model and randomize either the matching, the worker sorting, or both
def randomize_results(model_results,randomized):
    
    results = copy.deepcopy(model_results)
    
    if randomized == "matching":
        
        # Randomize the matching fun from the OT problem
        results['ot']['matching_fun'] = np.random.permutation(results['ot']['matching_fun'])
        firms_random = sm.get_firm_info(results['ot'],results['types'],results['config'])
        results['firms'] = firms_random
        
    if randomized == "sorting":
        pass

        
    if randomized == "all":
        pass
        
    return results



