import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse

# SMALL_SIZE = 18
# MEDIUM_SIZE = 15
BIGGER_SIZE = 12

plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels


# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type=str) ## CascadeBandit/SetCover/
parser.add_argument('exp_type', type=str) ## Reward/Regret/Cost/Rate

args = parser.parse_args()

exp = []

# if args.exp_type == 'Regret':
#     y_label = args.exp_type
#     title = "Cumulative Regret"

# if args.exp_type == 'Reward':
#     y_label = args.exp_type
#     title = "Average Reward"

if args.exp_type == 'Cost':
    y_label = args.exp_type
    title = "Total Cost"

if args.exp_type == 'Rate':
    y_label = 'Count'
    title = "Target Arm Trigger Rate"

exp_num = 0

while os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + str(exp_num) + '.csv')):
    data = pd.read_csv(os.path.join('./SimulationResults', args.exp_name, args.exp_type + str(exp_num) + '.csv'))#.iloc[:40000, :]
    exp.append(data)
    exp_num += 1

df = pd.concat(exp, axis=0, ignore_index=True)[::-1]

cols = df.columns.tolist()
cols = cols[::-1]
df = df[cols]

grouped_df_min = df.groupby(["Time(Iteration)"]).min()
grouped_df_max = df.groupby(["Time(Iteration)"]).max()
grouped_df_mean = df.groupby(["Time(Iteration)"]).mean()
grouped_df_std = df.groupby(["Time(Iteration)"]).std()

quant_num = 0.3
grouped_df_quantile_min = df.groupby(["Time(Iteration)"]).quantile(quant_num)
grouped_df_quantile_max = df.groupby(["Time(Iteration)"]).quantile(1-quant_num)


# labels = {'Randomized CUCB_Attack': 'Random Target', 'CUCB_Attack': 'Fixed Target', 'CascadeUCB-V-Attack': 'CascadeUCB-V', 'CascadeUCB1-Attack': 'CascadeUCB1', 'CascadeKLUCB-Attack': 'CascadeKLUCB'}

fig, ax = plt.subplots()
ax.ticklabel_format(style='sci', useOffset=True, scilimits=(0, 0))

colors = list(mcolors.TABLEAU_COLORS.keys())
cols = list(grouped_df_mean.columns)

for c in range(len(cols)):
    plt.plot(range(grouped_df_mean.shape[0]), grouped_df_mean[cols[c]], label=cols[c], color=colors[c])
    plt.fill_between(grouped_df_std.index, (grouped_df_mean[cols[c]] - grouped_df_std[cols[c]]).clip(0, None), grouped_df_mean[cols[c]] + grouped_df_std[cols[c]], color=colors[c], alpha=0.2)

    

print("plotting")
plt.xlabel('Iterations')
plt.ylabel(y_label)
plt.legend(loc="upper left")
plt.title(title)
plt.grid()

t = ax.yaxis.get_offset_text()
t.set_x(-0.05)

plt.tight_layout()

print("saving")
plt.savefig(os.path.join('./SimulationResults', args.exp_name, args.exp_type + '.png'))
# plt.show()

