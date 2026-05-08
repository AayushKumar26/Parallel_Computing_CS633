import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('timing_data.csv')

sns.set_theme(style='darkgrid')
sns.catplot(data=df, x='processes', y='time', kind='box', hue='data_size', height=7, aspect=2)

plt.subplots_adjust(top=0.95)

plt.title('Execution Time vs Process Count')
plt.xlabel('Processes(P)')
plt.ylabel('Time(T)')
plt.savefig('boxplot.png', dpi=600)
plt.show()
