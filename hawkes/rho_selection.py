import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('results.csv')
df.sort_values(by='delta')
intervention_factor = df['delta'].values
cost = df['cost'].values
cost_conf_int_left = df['cost_conf_int_left']
cost_conf_int_right = df['cost_conf_int_right']
deaths = np.ceil(df['deaths'].values).astype(int)
deaths_conf_int_left = df['deaths_conf_int_left']
deaths_conf_int_right = df['deaths_conf_int_right']
_, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].fill_between(
    x=intervention_factor,
    y1=deaths_conf_int_left,
    y2=deaths_conf_int_right,
    color='lightblue'
    )
ax[0].plot(intervention_factor, deaths, marker='o')
ax[0].set_xticks(intervention_factor)
ax[0].set_xlabel('Intervention factor')
ax[0].set_ylabel('Total deaths')
ax[0].set_title('Number of deaths in function of the intervention factor')
ax[0].axvline(x=32, color='red', linestyle='dashed')
ax[0].axline((0, 20000,), (2, 20000), color='black', linestyle='dotted')
ax[0].legend(('.95 confidence interval', 'Number of deaths', 'Choice for the intervention factor 32', 'Limit for the number of deaths'))
ax[1].fill_between(
    x=intervention_factor,
    y1=cost_conf_int_left,
    y2=cost_conf_int_right,
    color='lightblue'
    )
ax[1].plot(intervention_factor, cost, marker='o')
ax[1].set_xticks(intervention_factor)
ax[1].set_xlabel('Intervention factor')
ax[1].set_ylabel('Total cost')
ax[1].set_title('Total cost in function of the intervention factor')
ax[1].axvline(x=32, color='red', linestyle='dashed')
ax[1].legend(('.95 confidence interval', 'Total cost', 'Choice for the intervention factor 32'))
plt.show()
