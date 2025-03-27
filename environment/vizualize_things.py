### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec

from specification import Environment, demand_function

files = ["TN_DDQN_0", "PPO_D_0", "PPO_C_0", "SAC_0"]

Price_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
Profit_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
Stock_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]

data = [Price_list, Stock_list, Profit_list]
ex = [(1, 2.5), (0, 30), (-5, 15)]

demand_params = Environment["demand_params"]
spros = demand_function(**demand_params)
p_NE, p_M, pi_NE, pi_M = spros.get_theory(Environment["c_i"])
inv_NE, inv_M = spros.distribution([p_NE]*Environment["n"])[0], spros.distribution([p_M]*Environment["n"])[0]

th = [(p_NE, p_M), (inv_NE, inv_M), (pi_NE, pi_M)]
# d = 0.5
# ex = [tuple([min(x) - d*(max(x) - min(x)), max(x) + d*(max(x) - min(x))]) for x in th[:2]]
# ex = ex + [(-15, 15)]
lab = ["price", "stock", "profit"]

# fig, axes = plt.subplots(3, len(files), figsize=(16, 8))
gs = gridspec.GridSpec(4, len(files), height_ratios=[1, 1, 1, 0.05])
fig = plt.figure(figsize=(20, 12))

for i in range(3):
    for j in range(len(files)):
        
        # ax = axes[i][j]
        ax = fig.add_subplot(gs[i, j])

        bins = 20
        mn = min(ex[i][0], ex[i][1])
        mx = max(ex[i][0], ex[i][1])
        xedges = np.linspace(mn, mx, bins + 1)
        yedges = np.linspace(mn, mx, bins + 1)

        # Построение гистограммы
        hist, xedges, yedges = np.histogram2d(
            data[i][j][:, 0], data[i][j][:, 1],
            bins=(xedges, yedges)
        )
        hist_normalized = hist / len(data[i][j])

        im = ax.imshow(
            hist_normalized.T,
            origin='lower',
            extent=[ex[i][0], ex[i][1], ex[i][0], ex[i][1]],
            # extent = [th[i][0], th[i][1], th[i][0], th[i][1]],
            cmap='YlOrRd',
            aspect='auto',
            # interpolation= 'bicubic'# 'bicubic'
            # vmin=0, vmax=0.5,
        )

        if i == 0:
            ax.set_title(f'{files[j][:-2].replace("_", "-")}')

        # Скрыть числовые метки на осях
        # ax.set_xticks([])
        # ax.set_yticks([])
        
        ax.scatter([th[i][0]], [th[i][0]], s=20, color='blue', edgecolor='black', zorder=5)
        ax.text(th[i][0] + 0.02 * (ex[i][1] - ex[i][0]), th[i][0], 'NE', color='blue', fontweight='bold')

        # ax.text(th[i][1] + 0.02, th[i][1], 'M', color='black')
        # ax.plot([th[i][1]], [th[i][1]], 'kx', markersize=4)
        ax.scatter([th[i][1]], [th[i][1]], s=20, color='#00CD00', edgecolor='black', zorder=5)
        ax.text(th[i][1] + 0.02 * (ex[i][1] - ex[i][0]), th[i][1], 'M', color='#00CD00', fontweight='bold')

        # Осевые метки
        ax.set_xlabel(f'Agent i {lab[i]}') # , fontsize= 8)
        ax.set_ylabel(f'Agent j {lab[i]}') # , fontsize= 8)

        if i == 2:
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

# Общая шкала цветов
# cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.01])  # [left, bottom, width, height]
# max_value = max([hist.max() / len(data[i][j]) for i in range(3) for j in range(len(files))])
# ticks = np.linspace(0, max_value, 11)
# cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks = [round(x, 2) for x in ticks]) # , pad=0.01
# cbar.set_label('Occurrence Ratio', fontsize= 10)

cbar_ax = fig.add_subplot(gs[3, :])
max_value = max([hist.max() / len(data[i][j]) for i in range(3) for j in range(len(files))])
ticks = np.linspace(0, max_value, 11)
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks = [round(x, 2) for x in ticks]) # , pad=0.01
cbar.set_label('Occurrence Ratio', fontsize= 10)
cbar_ax.set_position([0.1, 0.12, 0.8, 0.03])

# Корректировка междуграфических отступов
# plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.tight_layout(rect=[0, 0.02, 1, 1], pad=1.5)
plt.savefig("./DRL_pricing/environment/simulation_results/platforms/plot_platform_None", dpi = 1000)

# plt.show()