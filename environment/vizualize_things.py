### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt

import matplotlib.pyplot as plt
import numpy as np
import os

from specification import Environment, demand_function

files = []

for filename in os.listdir("./DRL_pricing/environment/simulation_results/"):
    if filename.endswith("0"):
        files.append(filename)

files = files[::-1]

Price_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
Profit_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
Stock_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]

data = [Price_list, Stock_list, Profit_list]
ex = [(1, 2.5), (0, 30), (-15, 15)]

demand_params = Environment["demand_params"]
spros = demand_function(**demand_params)
p_NE, p_M, pi_NE, pi_M = spros.get_theory(Environment["c_i"])
inv_NE, inv_M = spros.distribution([p_NE]*Environment["n"])[0], spros.distribution([p_M]*Environment["n"])[0]

th = [(p_NE, p_M), (inv_NE, inv_M), (pi_NE, pi_M)]
# d = 0.5
# ex = [tuple([min(x) - d*(max(x) - min(x)), max(x) + d*(max(x) - min(x))]) for x in th]

fig, axes = plt.subplots(3, 3, figsize=(20, 12))

for i in range(3):
    for j in range(3):
        
        ax = axes[i][j]

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
        )

        if i == 0:
            ax.set_title(f'{files[j][:-2].replace("_", "-")}')

        # Скрыть числовые метки на осях
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Добавляем синий жирный кружочек для "Nash"
        ax.scatter([th[i][0]], [th[i][0]], s=20, color='blue', edgecolor='black', zorder=5)
        ax.text(th[i][0] + 0.01, th[i][0], 'NE', color='blue')

        ax.text(th[i][1] + 0.01, th[i][1], 'M', color='black')
        ax.plot([th[i][1]], [th[i][1]], 'kx', markersize=5)

        # Осевые метки
        # ax.set_xlabel('Agent i Price')
        # ax.set_ylabel('Agent j Price')

# Общая шкала цветов
cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Occurrence Ratio')

# Корректировка междуграфических отступов
# plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.show()