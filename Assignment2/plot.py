import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('timings.csv')

processes = [32, 48, 64, 96]
sizes     = [120, 240]
colors    = {120: '#4C72B0', 240: '#DD8452'}
labels    = {120: 'nx=ny=nz=120', 240: 'nx=ny=nz=240'}

fig, ax = plt.subplots(figsize=(9, 6))

group_width = 1.0
box_width   = 0.30
offsets     = {120: -0.22, 240: +0.22}

x_ticks = np.arange(len(processes))          # 0,1,2,3

bp_handles = []
for nx in sizes:
    positions = x_ticks + offsets[nx]
    data_per_p = [df[(df['P'] == p) & (df['nx'] == nx)]['time'].values
                  for p in processes]

    bp = ax.boxplot(data_per_p, positions=positions, widths=box_width, patch_artist=True, notch=False,
                    boxprops=dict(facecolor=colors[nx], alpha=0.75), 
                    medianprops=dict(color='black', linewidth=2), whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4), 
                    flierprops=dict(marker='o', markersize=5, markerfacecolor=colors[nx], linestyle='none'))
    
    bp_handles.append(plt.Rectangle((0,0),1,1, facecolor=colors[nx], alpha=0.75, label=labels[nx]))

ax.set_xticks(x_ticks)
ax.set_xticklabels([f'P={p}' for p in processes], fontsize=12)
ax.set_xlabel('Number of Processes (P)', fontsize=13)
ax.set_ylabel('Time (seconds)', fontsize=13)
ax.set_title('Execution Time vs Number of Processes\n(5 runs per configuration)', fontsize=14)
ax.legend(handles=bp_handles, fontsize=11, loc='upper left')
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('boxplot.png', dpi=150)
print("Saved boxplot.png")
