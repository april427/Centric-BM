# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
log_dir = 'RL_mimo_ofdm/PPO_2ue16Nt_test/'
# %%

df = pd.read_csv(log_dir+"throughput.csv")
thp_cdf1 = np.sort(df['baseline'])
yvals1 = np.arange(len(thp_cdf1))/float(len(thp_cdf1)-1)
thp_cdf2 = np.sort(df['benchmark'])
yvals2 = np.arange(len(thp_cdf2))/float(len(thp_cdf2)-1)
figure_handle = plt.figure()
ax = figure_handle.add_subplot()
ax.plot(thp_cdf1, yvals1, '-')
ax.plot(thp_cdf2, yvals2, '--')
ax.legend(['PPO','Genie_AoD'])
ax.set_xlabel('Throughput')
ax.set_ylabel('CDF')

# %%
df1 = pd.read_csv(log_dir+"latency.csv")
latency1 = df1['baseline']
df2 = pd.read_csv(log_dir + "latency_benchmark.csv")   
latency2 = df2['benchmark']

figure_handle = plt.figure()
ax2 = figure_handle.add_subplot()
ax2.hist(latency1, bins=10, alpha=0.7, density = True, label='PPO')
ax2.hist(latency2, bins=10, alpha=0.7, density = True, label = 'Genie_AoD')
ax2.legend()
ax2.set_xticks(range(5))
ax2.set_xticklabels(range(5))
ax2.set_xlabel('Latency')
# %%
