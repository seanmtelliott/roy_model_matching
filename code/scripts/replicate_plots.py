# Replicate Bloom plots

# Set working directory and import libraries
import pandas as pd, matplotlib.pyplot as plt, os
os.chdir(os.path.dirname(__file__))
os.chdir('../..')

# Read in the data (these were cleaned/organized in R)
ineq_data = pd.read_csv(os.path.join(os.getcwd(),'data', 'input','bloom_data_1983_2013.csv'))
ineq_data = ineq_data[ineq_data.percentile != 100]

# Generate the plot
ticks = ineq_data['percentile']

plt.plot(ticks,ineq_data['within_ineq'],label = "Within Firm", color="g")
plt.plot(ticks,ineq_data['firm_ineq'], label = "Firms", color="r")
plt.plot(ticks,ineq_data['ind_ineq'], label = "Individuals", color = "royalblue")
plt.legend(loc="upper left")
plt.xlabel("Percentile")
plt.ylabel("Diff. of natural log")
plt.savefig(os.path.join(os.getcwd(),'data', 'output','test_plots','bloom_data_1983_2013.png'))