### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import stats

from specification import Environment, demand_function

SHOW = False
CREATE = False
SQUEEZE = False
KS = False
VAR = False
pl_list = ["None", "fixed" ,"dynamic"]
num = "2"
platform = pl_list[int(num)]
DIFF_PL = (num == "2")
files = ["TN_DDQN_0", "PPO_D_0", "PPO_C_0", "SAC_0"]
files = [x.replace("_0", f"_{num}_0") for x in files]
names = ["TN-DDQN", "PPO-D", "PPO-C", "SAC"]

start_collusion = """\\bgroup
\def\arraystretch{1.25}
\begin{table}[H]
\caption{Индексы сговора (по усред. последним 5\% итераций)}
\label{tables:platforms_None}
\begin{center}"""

# \caption{Расчетные статистики теста Колмогорова-Смирнова согласованности распределений параметров равновесия по 5\% последних итераций}
start_ks = """\\bgroup
\def\arraystretch{1.25}
\begin{table}[H]
\caption{Расчетные статистики U-теста для распределений параметров равновесия по 5\% последних итераций}
\label{tables:ks_fixed}
\begin{center}"""

end_collusion = """\end{center}
\end{table}
\egroup"""

if KS:
    files = ["TN_DDQN_0", "PPO_D_0", "PPO_C_0", "SAC_0"][:-1]
    files = [x.replace("_0", "_0_0") for x in files]
    Price_zero = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
    Profit_zero = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
    Stock_zero = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]
    data = np.array([Price_zero, Stock_zero, Profit_zero])
    data_old = data.transpose(1, 0, 2, 3)

    files = ["TN_DDQN_0", "PPO_D_0", "PPO_C_0", "SAC_0"][:-1]
    files = [x.replace("_0", f"_{num}_0") for x in files]
    Price_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
    Profit_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
    Stock_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]
    data = np.array([Price_list, Stock_list, Profit_list])
    data_new = data.transpose(1, 0, 2, 3)

    dic = {names[i]:[] for i in range(len(files))}

    for i in range(len(files)):
        for j in range(3):
            # a = stats.ks_2samp(data_new[i][j].flatten(), data_old[i][j].flatten())
            b = stats.mannwhitneyu(data_new[i][j].flatten(), data_old[i][j].flatten(), alternative='less')
            c = stats.mannwhitneyu(data_new[i][j].flatten(), data_old[i][j].flatten(), alternative="greater")
            d = stats.mannwhitneyu(data_new[i][j].flatten(), data_old[i][j].flatten(), alternative="two-sided")
            if (d.pvalue < 0.1) and (b.pvalue < c.pvalue):
                a = b
                sign = "<"
            elif (d.pvalue < 0.1) and (b.pvalue > c.pvalue):
                a = c
                sign = ">"
            elif (d.pvalue >= 0.1):
                a = d
                sign = "0"
            if i in [0] and j == 0:
                print(b.pvalue, c.pvalue, d.pvalue)
                res1 = stats.ecdf(data_new[i][j].flatten())
                res2 = stats.ecdf(data_old[i][j].flatten())
                ax = plt.subplot()
                res1.cdf.plot(ax, color = "blue")
                res2.cdf.plot(ax, color = "orange")
                plt.show()
            dots = "*" * int(a.pvalue < 0.1)  + "*" * int(a.pvalue < 0.05)  + "*" * int(a.pvalue < 0.01)
            text = "$"* int(len(dots) > 0) + str(round(a.statistic, 2)) + ("^{" + dots + "}") * int(len(dots) > 0)
            text = "\makecell[c]{ " + text + ("_{" + sign + "} $") * int(len(dots) > 0) +"\\\\[1ex] }" # + (" \\\\ (" + sign + ") ") * int(len(dots) > 0)
            dic[names[i]].append(text)
    
    df = pd.DataFrame(dic).T
    # df.columns=["$KS_{price}$", "$KS_{inv}$", "$KS_{\pi}$"]
    df.columns=["$U_{price}$", "$U_{inv}$", "$U_{\pi}$"]
    print(start_ks)
    print(df.to_latex())
    print(end_collusion)


if CREATE:
    dic = {names[i]:[] for i in range(len(files))}
    for i, x in enumerate(files):
        with open(f"./DRL_pricing/environment/simulation_results/{x}/summary.txt", "r+", encoding="utf-8") as f:
            A = f.readlines()
        mas = [dd.strip().replace("%", "\%").split(":")[1].strip() for dd in A[-3:]]
        dic[names[i]] = mas
    df = pd.DataFrame(dic).T
    df.columns=["$\Delta_{price}$", "$\Delta_{inv}$", "$\Delta_{\pi}$"]
    print(start_collusion)
    print(df.to_latex())
    print(end_collusion)


if SHOW:
    assert platform in ["None", "fixed", "dynamic"]

    Price_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
    Profit_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
    Stock_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]

    data = [Price_list, Stock_list, Profit_list]
    ex = [(Environment["p_inf"], Environment["p_sup"]),
          (0, Environment["demand_params"]["C"]),
          (-5, 15)]

    demand_params = Environment["demand_params"]
    spros = demand_function(**demand_params)
    gamma = Environment["gamma"]
    theta_d = Environment["theta_d"]
    p_NE, p_M, pi_NE, pi_M = spros.get_theory(Environment["c_i"], gamma, theta_d)
    inv_NE, inv_M = spros.distribution([p_NE]*Environment["n"])[0], spros.distribution([p_M]*Environment["n"])[0]

    th = [(p_NE, p_M), (inv_NE, inv_M), (pi_NE, pi_M)]
    # d = 0.5
    # ex = [tuple([min(x) - d*(max(x) - min(x)), max(x) + d*(max(x) - min(x))]) for x in th[:2]]
    # ex = ex + [(-15, 15)]
    lab = ["price", "stock", "profit"]
    y_lab = ["Цены", "Запасы", "Прибыль"]

    # fig, axes = plt.subplots(3, len(files), figsize=(16, 8))
    gs = gridspec.GridSpec(4, len(files), height_ratios=[1, 1, 1, 0.02],
                           hspace=0.2, wspace=0.2)
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
                ax.set_title(names[j], fontsize= 20)
            
            if j == 0:
                ax.set_ylabel(y_lab[i], fontsize= 16)

            # Скрыть числовые метки на осях
            # ax.set_xticks([])
            # ax.set_yticks([])
            
            ax.scatter([th[i][0]], [th[i][0]], s=30, color='blue', edgecolor='black', zorder=5)
            ax.text(th[i][0] + 0.02 * (ex[i][1] - ex[i][0]), th[i][0], 'NE', color='blue', fontweight='bold', fontsize= 14)


            # ax.text(th[i][1] + 0.02, th[i][1], 'M', color='black')
            # ax.plot([th[i][1]], [th[i][1]], 'kx', markersize=4)
            ax.scatter([th[i][1]], [th[i][1]], s=30, color='#00CD00', edgecolor='black', zorder=5)
            ax.text(th[i][1] + 0.02 * (ex[i][1] - ex[i][0]), th[i][1], 'M', color='#00CD00', fontweight='bold', fontsize= 14)

            # Осевые метки
            # ax.set_xlabel(f'Agent i {lab[i]}') # , fontsize= 8)
            # ax.set_ylabel(f'Agent j {lab[i]}') # , fontsize= 8)

            if i == 2:
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # Общая шкала цветов
    # cbar_ax = fig.add_axes([0.1, 0.04, 0.8, 0.01])  # [left, bottom, width, height]
    # max_value = max([hist.max() / len(data[i][j]) for i in range(3) for j in range(len(files))])
    # ticks = np.linspace(0, max_value, 11)
    # cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks = [round(x, 2) for x in ticks]) # , pad=0.01
    # cbar.set_label('Частота', fontsize= 10)

    cbar_ax = fig.add_subplot(gs[3, :])
    max_value = max([hist.max() / len(data[i][j]) for i in range(3) for j in range(len(files))])
    ticks = np.linspace(0, max_value, 11)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks = [round(x, 2) for x in ticks]) # , pad=0.01
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Частота', fontsize= 16)
    cbar_ax.set_position([0.1, 0.1, 0.8, 0.01])

    # Корректировка междуграфических отступов
    # plt.tight_layout(rect=[0, 0.05, 1, 1])
    # plt.tight_layout(rect=[0, 0.02, 1, 1], pad=1.5)
    plt.savefig(f"./DRL_pricing/environment/simulation_results/platforms/plot_platform_{platform}", dpi = 200, bbox_inches='tight')

    plt.show()


if SQUEEZE:
    assert platform in ["None", "fixed", "dynamic"]

    Price_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy").flatten() for x in files]
    Profit_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy").flatten() for x in files]
    Stock_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy").flatten() for x in files]
    if DIFF_PL:
        Platform_history = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Platform_history.npy").flatten() for x in files]
        Platform_actions = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Platform_actions.npy").flatten() for x in files]

    data = [Price_list, Stock_list, Profit_list]

    demand_params = Environment["demand_params"]
    spros = demand_function(**demand_params)
    gamma = Environment["gamma"]
    theta_d = Environment["theta_d"]
    p_NE, p_M, pi_NE, pi_M = spros.get_theory(Environment["c_i"], gamma, theta_d)
    inv_NE, inv_M = spros.distribution([p_NE]*Environment["n"])[0], spros.distribution([p_M]*Environment["n"])[0]

    th = [(p_NE, p_M), (inv_NE, inv_M), (pi_NE, pi_M)]
    y_lab = ["Цены", "Запасы", "Прибыль"]

    gs = gridspec.GridSpec(4, len(files), height_ratios=[1, 1, 1, 0.02],
                           hspace=0.35, wspace=0.2)
    fig = plt.figure(figsize=(20, 12))

    for i in range(3):
        for j in range(len(files)):
            
            ax = fig.add_subplot(gs[i, j])

            counts, bins = np.histogram(data[i][j])
            counts = counts / len(data[i][j])

            ax.hist(bins[:-1], bins, weights=counts, color = "#8B8B83") # , edgecolor='black'

            if i == 0:
                ax.set_title(names[j], fontsize= 20)

            ax.axvline(x=th[i][1], color='#00CD00', linestyle='--', linewidth=2, label = "M")
            ax.axvline(x=th[i][0], color='blue', linestyle='--', linewidth=2, label = "NE")

            if i == 0 and j == 0:
                ax.legend(fontsize=14, loc = "upper right")
            
            if j == 0:
                ax.set_ylabel(y_lab[i], fontsize=16)

            lower_bound = min(th[i][0], th[i][1])
            upper_bound = max(th[i][0], th[i][1])
            total_count = len(data[i][j])
            within_interval = np.sum((data[i][j] >= lower_bound) & (data[i][j] <= upper_bound))
            percentage_within = (within_interval / total_count) * 100

            ax.text(
                0.5, -0.15, f"{percentage_within:.1f}%", 
                transform=ax.transAxes, fontsize=18, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                color= "red",
            )

            mx = max(bins)
            mn = min(bins)

            if i == 2:
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

    plt.savefig(f"./DRL_pricing/environment/simulation_results/platforms/dencies_platform_{platform}", dpi = 200, bbox_inches='tight')

    plt.show()

    if DIFF_PL:
        data = [Platform_actions, Platform_history]

        gs = gridspec.GridSpec(4, len(files), height_ratios=[1, 1, 1, 0.02],
                            hspace=0.35, wspace=0.2)
        fig = plt.figure(figsize=(20, 12))

        y_lab = ["Коэфф. бустинга, %", "Прибыль"]

        for i in range(2):
            for j in range(len(files)):
                
                ax = fig.add_subplot(gs[i, j])

                counts, bins = np.histogram(data[i][j])
                counts = counts / len(data[i][j])

                ax.hist(bins[:-1], bins, weights=counts, color = "#8B8B83") # , edgecolor='black'

                if i == 0:
                    ax.set_title(names[j], fontsize= 20)

                if j == 0:
                    ax.set_ylabel(y_lab[i], fontsize=16)

        plt.savefig(f"./DRL_pricing/environment/simulation_results/platforms/dencies_platform_{platform}_PL", dpi = 200, bbox_inches='tight')

        plt.show()


def bootstrap_test(sample1, sample2, stat_func=np.mean, n_boot=10000):
    observed_diff = stat_func(sample1) - stat_func(sample2)
    pooled = np.concatenate([sample1, sample2])
    n1, n2 = len(sample1), len(sample2)
    
    boot_diffs = []
    for _ in range(n_boot):
        boot1 = np.random.choice(pooled, size=n1, replace=True)
        boot2 = np.random.choice(pooled, size=n2, replace=True)
        diff = stat_func(boot1) - stat_func(boot2)
        boot_diffs.append(diff)
    p_value = (np.abs(boot_diffs) >= np.abs(observed_diff)).mean()
    
    return observed_diff, p_value


# if VAR:
#     assert platform in ["None", "fixed", "dynamic"]

#     files = ["TN_DDQN_0", "PPO_D_0", "PPO_C_0", "SAC_0"][:-1]
#     files = [x.replace("_0", "_0_0") for x in files]
#     Price_zero = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
#     Profit_zero = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
#     Stock_zero = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]
#     data = np.array([Price_zero, Stock_zero, Profit_zero])
#     data_old = data.transpose(1, 0, 2, 3)

#     files = ["TN_DDQN_0", "PPO_D_0", "PPO_C_0", "SAC_0"][:-1]
#     files = [x.replace("_0", f"_{num}_0") for x in files]
#     Price_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Price_history.npy") for x in files]
#     Profit_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Profit_history.npy") for x in files]
#     Stock_list = [np.load(f"./DRL_pricing/environment/simulation_results/{x}/Stock_history.npy") for x in files]
#     data = np.array([Price_list, Stock_list, Profit_list])
#     data_new = data.transpose(1, 0, 2, 3)

#     demand_params = Environment["demand_params"]
#     spros = demand_function(**demand_params)
#     gamma = Environment["gamma"]
#     theta_d = Environment["theta_d"]
#     p_NE, p_M, pi_NE, pi_M = spros.get_theory(Environment["c_i"], gamma, theta_d)
#     inv_NE, inv_M = spros.distribution([p_NE]*Environment["n"])[0], spros.distribution([p_M]*Environment["n"])[0]

#     th = [(p_NE, p_M), (inv_NE, inv_M), (pi_NE, pi_M)]

#     print(data_old[0][2][:5])

#     print(bootstrap_test(data_old[0][2].flatten(), data_new[0][2].flatten()))