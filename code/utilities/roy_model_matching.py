################################################################################
# Sean Elliott - March 2024
#
# Simulations for "A generalization of a model of Roy for partition of labour force via matching and occupational choice"
#
# This is just the example of recreating the simulations already performed
# The code in here will be generalized to allow for any underlying skill distribution
# and other relevant parameters.
# I will make a note anywhere that a parameterization/generalization is needed.
################################################################################

# Import libraries
import sys, matplotlib.pyplot as plt, logging
logging.getLogger().setLevel(logging.CRITICAL)
sys.path.append('code/utilities')
import sim_methods as sm

## PERFORM SIMULATION

def model_sim(size,dist,dist_params,revenue_params):
    
    # Set the config file
    
    config = sm.modify_config(size, dist, dist_params, revenue_params)
    
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
    
    results = {}
    results['ot'] = ot_results
    results['types'] = types
    results['config'] = config
    
    return results

###############################################################################

## GENERATE PLOTS

def gen_plots(results,labels,output_path):
    
    # Put the plots together
    fig, axes = plt.subplots(2,2)
    
    #config = results[0]['config']
    
    
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
        wage_key = results[i]['ot']['wage_key']
        wage_sec = results[i]['ot']['wage_sec']
    
        # Wages
        if i < len(labels)-1:
            wage_lab = ["_Hidden","_Hidden"]
        elif i==len(labels)-1:
            wage_lab = ["key","secondary"]
            
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
        matching_fun = [results[i]['ot']['ot_mat'][k].argmax()/(num_types-1) for k in range(num_types)] 
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
    if len(labels) == 3:
        fig.legend(lines, labels, loc="lower center",ncol=(len(labels)),  prop = { "size": 7.5 })
    if len(labels) == 5:
        fig.legend(lines, labels, bbox_to_anchor=(0.44, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })

    
    plt.tight_layout()
    plt.savefig(output_path)

    
    return 

###############################################################################


## GENERATE WORKER DF
def gen_worker_df(ot_results,worker_df):
    
    pi_new = ([-x for x in ot_results[1]['u']])
    w_new = ([-x for x in ot_results[1]['v']])
    k_types = worker_df['k'].unique()
    s_types = worker_df['s'].unique()
    
    pi_new_df = pd.DataFrame(np.column_stack((k_types,pi_new,)), columns=['k','wage_key_new'])
    w_new_df = pd.DataFrame(np.column_stack((s_types,w_new)),columns=['s','wage_sec_new'])
    
    # Update the wages
    worker_df_new = worker_df.merge(pi_new_df, on='k', how='left').merge(w_new_df, on='s', how='left')
    worker_df_new['wage_key'] = worker_df_new['wage_key_new']
    worker_df_new['wage_sec'] = worker_df_new['wage_sec_new']
    worker_df_new = worker_df_new.drop('wage_key_new', axis=1).drop("wage_sec_new", axis=1)
    
    return worker_df_new

