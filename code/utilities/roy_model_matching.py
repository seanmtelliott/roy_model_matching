################################################################################
# Sean Elliott - March 2024
#
# Simulations for "A generalization of a model of Roy for partition of labour force via matching and occupational choice"
#
# These are the callable functions from the perspective of the user.
# It contains the main functionality:
# 1. Perform simulation
# 2. Generate plots based on simulation results
################################################################################

# Import libraries
import sys, matplotlib.pyplot as plt, logging, numpy as np, statistics, sim_methods as sm, os, statistics as stats
import seaborn as sns, pandas as pd
from scipy.stats import norm
logging.getLogger().setLevel(logging.CRITICAL)
sys.path.append('code/utilities')

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
   # fig, axes = plt.subplots(2,2)
    
    fig = plt.figure(figsize=(5.5, 3.5), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    
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
            
        ax1.plot(types_key,wage_key,color=colors[i],label=wage_lab[0])
        ax1.plot(types_sec,wage_sec,linestyle="dotted",color=colors[i],label=wage_lab[1])
        ax1.legend(loc='lower right',fontsize="7",ncol=2)
        ax1.set_title("Ln wage by type")
        ax1.set_xlabel('Skill level')
        ax1.set_ylabel('Ln wage')

        # Separating function
        #wage_differential = [[wage_key[k]-wage_sec[s] for s in range(num_types)] for k in range(num_types)]
        wage_differential_abs = [[abs(wage_key[k]-wage_sec[s]) for s in range(num_types)] for k in range(num_types)]
        sep_function = [wage_differential_abs[k].index(min(wage_differential_abs[k]))/(num_types-1) for k in range(num_types)]
        ax3.plot(types_key,sep_function,color=colors[i],label = labels[i])
        ax3.set_title("Separating function")
        ax3.set_xlabel('k')
        ax3.set_ylabel('s')
    
        # Matching function
        matching_fun = results[i]['ot']['matching_fun']
        ax2.plot(types_key,matching_fun,color=colors[i],label = labels[i])
        ax2.set_title("Matching function")
        ax2.set_xlabel('k')
        ax2.set_ylabel('s')
    
        # Wage inequality
        # ineq_function = [wage_differential[k][int(matching_fun[k]*(num_types-1))] for k in range(num_types)]
        # axes[1,1].plot(types_key,ineq_function,color=colors[i],label = labels[i])
        # axes[1,1].set_title("Wage inequality in matches")
        # axes[1,1].set_xlabel('k')
        # axes[1,1].set_ylabel('Wage difference')
        
    lines_labels = [fig.axes[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Position the legend differently depending on how many labels we have 
    # TODO: generalize this also
    if len(labels) == 3:
        fig.legend(lines, labels, bbox_to_anchor=(0.25, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })
    if len(labels) == 5:
        fig.legend(lines, labels, bbox_to_anchor=(0.44, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })

    
    plt.tight_layout()
    plt.savefig(output_path)

    
    return 

###############################################################################


## RANDOMIZE RESULTS
# Generate output identical to that of the model but everything is randomized
def randomize_results(size,dist,dist_params,revenue_params,tolerance=0.01):
    
    # Set the config file
    config = sm.modify_config(size, dist, dist_params, revenue_params,tolerance)
    
    #Display to the user the simulation being run
    sm.sim_params(config)
    
    # Get type information
    types = sm.gen_workers(config)

    # Get wage info (here it is just their marginal contribution to revenue)
    wages = sm.set_init_wage(types.copy(),config,random=True)
    
    # Randomize the matching (just sort the sec skills randomly and they match to key role monotonically)
    matching_fun = np.random.permutation(types['sec'])
    
    # Format this like the OT results would be 
    ot_like_results = {}
    ot_like_results['wage_key']=wages['key']
    ot_like_results['wage_sec']=wages['sec']
    ot_like_results['matching_fun']=matching_fun
    
    # Get firm info
    firms = sm.get_firm_info(ot_like_results,types,config)
    
    results = {}
    results['config'] = config
    results['firms'] = firms
    results['ot'] = ot_like_results
    results['types'] = types
    
    return results

###############################################################################


## GENERATE INEQUALITY RESULTS
# Results is a dictionary containing model output objects, wage inequality comparisons are made across these model results
# We can compute three different forms of inequality:
#   1. within firms (difference between pair worker wages)
#   2. across firms (difference between average output by firm)
#   3. across individuals (do they high-skill individuals earn comparatively more under sorting/matching?)
def plot_inequality(results,labels,output_path):
    
    # Get the scenario names
    scen1 = labels[0]
    scen2 = labels[1]
          
    # Need to weight everything by the counts of individuals in the generated sample
    weighted_firm1 = sm.get_pop_weights(results[scen1])
    weighted_firm2 = sm.get_pop_weights(results[scen2])
    
    
    # Individuals

    ind1 = statistics.quantiles(np.append(np.array(weighted_firm1['log_wage_sec']),np.array(weighted_firm1['log_wage_key'])),n=200)
    ind2 = statistics.quantiles(np.append(np.array(weighted_firm2['log_wage_sec']),np.array(weighted_firm2['log_wage_key'])),n=200)

    
    # Firms
    
    firm1 = statistics.quantiles(weighted_firm1['firm_output'],n=200)
    firm2 = statistics.quantiles(weighted_firm2['firm_output'],n=200)
    
    # Individual/firm
    
    within1 = statistics.quantiles(np.array(weighted_firm1['resid_key'],weighted_firm1['resid_sec']),n=200)
    within2 = statistics.quantiles(np.array(weighted_firm2['resid_key'],weighted_firm2['resid_sec']),n=200)

    # Put the plot together and write it to the output path
    ticks = np.arange(199)/2
    
    plt.plot(ticks,np.subtract(within2,within1),label = "Within Firm", color="g")
    plt.plot(ticks,np.subtract(firm2,firm1), label = "Firms", color="r")
    plt.plot(ticks,np.subtract(ind2,ind1), label = "Individuals", color = "royalblue")
    plt.legend(loc="upper right")
    plt.xlabel("Percentile")
    plt.ylabel("Diff. of natural log")
    plt.savefig(output_path)
    
    return

def plot_ineq_cross_sect(results,output_path): 

    weighted_firm = sm.get_pop_weights(results)
    weighted_firm['rank'] = np.arange(len(weighted_firm))/len(weighted_firm)
    weighted_firm['ineq'] = weighted_firm['log_wage_key'] - weighted_firm['log_wage_sec']
    plt.plot( weighted_firm['rank'],weighted_firm['ineq'])
    plt.ylabel("Diff. of log wages of key/sec")
    plt.xlabel("Percentile")
    plt.savefig(output_path)
        
    return

###############################################################################


## GENERATE IDENTIFICATION PLOTS
# We modify the skill distribution and analyze how the observable wage dist/matching function change and the unobserved separating function

def plot_indentification(results,labels,output_path,file_name):
    
    for j in range(len(labels)):
        # Contour plot of skill dist (Need to save these separately)
        img_name = os.path.join(output_path,'contour'+labels[j]+'.png')
        worker_sample = pd.DataFrame(results[j]['types']['workers'],columns=['k','s']).sample(n=1000)
        g1 = sns.jointplot(data=worker_sample,x="k",y="s",kind="kde")
        g1.fig.suptitle(labels[j])
        g1.savefig(img_name)
    
    fig = plt.figure(figsize=(5.5, 3.5), layout="constrained")
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    
    if len(labels) == 2:
        colors = ['grey','brown']
    if len(labels) == 3:
        colors = ['grey','brown','k']
    if len(labels) == 5:
        colors = ['grey','brown','k','blue','green']
    
    for i in range(len(labels)):
    
        # Get types
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
        

        # Separating function
        #wage_differential = [[wage_key[k]-wage_sec[s] for s in range(num_types)] for k in range(num_types)]
        wage_differential_abs = [[abs(wage_key[k]-wage_sec[s]) for s in range(num_types)] for k in range(num_types)]
        sep_function = [wage_differential_abs[k].index(min(wage_differential_abs[k]))/(num_types-1) for k in range(num_types)]
        ax3.plot(types_key,sep_function,color=colors[i],label = labels[i])
        ax3.set_title("Separating fun")
        ax3.set_xlabel('k')
        ax3.set_ylabel('s')
    
        # Matching function
        matching_fun = results[i]['ot']['matching_fun']
        ax2.plot(types_key,matching_fun,color=colors[i],label = labels[i])
        ax2.set_title("Matching fun")
        ax2.set_xlabel('k')
        ax2.set_ylabel('s')

        #Wages
        ax1.plot(types_key,wage_key,color=colors[i],label=wage_lab[0])
        ax1.plot(types_sec,wage_sec,linestyle="dotted",color=colors[i],label=wage_lab[1])
        ax1.legend(loc='lower right',fontsize="7",ncol=2)
        ax1.set_title("Ln wage by type")
        ax1.set_xlabel('Skill level')
        ax1.set_ylabel('Ln wage')
        
    lines_labels = [fig.axes[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Position the legend differently depending on how many labels we have 
    # TODO: generalize this also
    if len(labels) == 3:
        fig.legend(lines, labels, bbox_to_anchor=(0.25, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })
    if len(labels) == 5:
        fig.legend(lines, labels, bbox_to_anchor=(0.44, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })

    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,file_name[0]))
    
    fig = plt.figure(figsize=(5.5, 3.5), layout="constrained")
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    
    for i in range(len(labels)):

        weighted_firms = sm.get_pop_weights(results[i])

        
        # Wages
        if i < len(labels)-1:
            wage_lab = ["_Hidden","_Hidden"]
        elif i==len(labels)-1:
            wage_lab = ["k","s"]
        
        
        # Conditional wage distribution
        
        ax1.plot(np.sort(weighted_firms['wage_key']), np.linspace(0, 1, len(weighted_firms['wage_key']), endpoint=False),color=colors[i],label=wage_lab[0])
        ax1.plot(np.sort(weighted_firms['wage_sec']), np.linspace(0, 1, len(weighted_firms['wage_key']), endpoint=False),linestyle="dotted",color=colors[i],label=wage_lab[1])
        ax1.legend(loc='lower right',fontsize="7",ncol=2)
        ax1.set_title("Wage distribution")
        ax1.set_xlabel('Wage')
        ax1.set_ylabel('F(y|x)')
        
        # Matching function
        
        wage_key = np.log(results[i]['ot']['wage_key'])
        wage_sec = np.log(results[i]['ot']['wage_sec'])
        
        ax2.plot(results[i]['firms']['wage_key'],results[i]['firms']['wage_sec'],color=colors[i],label = labels[i])
        ax2.set_title("Observed matches")
        ax2.set_xlabel('wage k')
        ax2.set_ylabel('wage s')
        
        # Firm revenue
        
        weighted_firm = sm.get_pop_weights(results[i])
        firm_revnue = statistics.quantiles(weighted_firm['firm_output'],n=200)
        ticks = np.arange(199)/2
        ax3.plot(ticks,firm_revnue,color=colors[i],label = labels[i])
        ax3.set_title("Firm revenue")
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('R(k,s)')
        
        fig.legend(lines, labels, bbox_to_anchor=(0.25, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })
        
    lines_labels = [fig.axes[1].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # Position the legend differently depending on how many labels we have 
    # TODO: generalize this also
    if len(labels) == 3:
        fig.legend(lines, labels, bbox_to_anchor=(0.25, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })
    if len(labels) == 5:
        fig.legend(lines, labels, bbox_to_anchor=(0.44, -0.45, 0.5, 0.5),ncol=(len(labels)),  prop = { "size": 7.5 })

    plt.tight_layout()
    plt.savefig(os.path.join(output_path,file_name[1]))
    return

def get_estimated_moments(results):
    
    weighted_firm = sm.get_pop_weights(results)
    
    pr_key_sel = 0.5

    mean_key = stats.mean(weighted_firm['wage_key'])
    mean_sec = stats.mean(weighted_firm['wage_sec'])

    var_key = stats.variance(weighted_firm['wage_key'])
    var_sec = stats.variance(weighted_firm['wage_sec'])

    weighted_firm["mean_dev3_key"] = (weighted_firm['wage_key']-mean_key)**3
    weighted_firm["mean_dev3_sec"]  = (weighted_firm['wage_sec']-mean_sec)**3

    skew_key = stats.mean(weighted_firm["mean_dev3_key"])
    skew_sec =  stats.mean(weighted_firm["mean_dev3_sec"])

    D = norm.ppf(pr_key_sel)
    lD = sm.inv_mills(D)
    lDneg = sm.inv_mills(-D)

    tau_k3 = (skew_key/(lD * (2*lD**2 + 3 * D * lD + D**2 - 1)))
    tau_k = np.sign(tau_k3) * (np.abs(tau_k3)) ** (1 / 3)
    tau_s3 = (skew_sec/(lDneg * (2*lDneg**2 - 3 * D * lDneg + D**2 - 1)))
    tau_s = np.sign(tau_s3) * (np.abs(tau_s3)) ** (1 / 3)
    mu_k = mean_key - tau_k * lD
    mu_s = mean_sec - tau_s * lDneg
    sigma2_k = var_key - tau_k**2 * ((-lD)*D - lD**2)
    sigma2_s = var_sec - tau_s**2 * ((lDneg)*D - lDneg**2)
    #sigma_ks = (-(mu_k**2)+2*mu_k*mu_s - mu_s**2 + sigma2_k*(D**2) + sigma2_s*(D**2))/(2*D**2)

    mean = [mu_k,mu_s]
    return mean