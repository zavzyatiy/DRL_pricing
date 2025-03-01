### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from firms_RL import epsilon_greedy, TQL
from specification import Environment, demand_function

### иницализация randomseed
# random.seed(42)
# torch.manual_seed(42)

### количество итераций внутри среды
T = Environment["T"]

### количество симуляций среды
ENV = Environment["ENV"]

# число фирм
n = Environment["n"]
# горизонт памяти платформы
m = Environment["m"]
# коэффициент дисконтирования
delta = Environment["delta"]
# склонность доверия рекламе
gamma = Environment["gamma"]
# издержки закупки товара
c_i = Environment["c_i"]
# издержки хранения остатков
h_plus = Environment["h_plus"]
# издержки экстренного дозаполнения
v_minus = Environment["v_minus"]
# вероятность оставить отзыв
eta = Environment["eta"]
# нижняя граница цены == MC
p_inf = Environment["p_inf"]
# верхняя граница цены == адеватность/монополия
p_sup = Environment["p_sup"]
# количество цен для перебора при
# дискретизации пространства возможных цен
arms_amo = Environment["arms_amo"]

# Какими цветами рисовать?
color = Environment["color"]
# Как усреднять прибыли: усреднять независимо
# или считать для усредненных цен
profit_dynamic = Environment["profit_dynamic"]
# Выводить итоговый график?
VISUALIZE = Environment["VISUALIZE"]
# Выводить теоретические величины для NE и M?
VISUALIZE_THEORY = Environment["VISUALIZE_THEORY"]
# Сохранить итоговоый график?
SAVE = Environment["SAVE"]
# С какой стороны отображать легенду на графиках
loc = Environment["loc"]

# Цены
prices = Environment["prices"]
MEMORY_VOLUME = Environment["firm_params"]["MEMORY_VOLUME"]
own = Environment["firm_params"]["own"]
ONLY_OWN = Environment["firm_params"]["ONLY_OWN"]

Price_history = []
Profit_history = []

demand_params = Environment["demand_params"]
spros = demand_function(**demand_params)
if VISUALIZE_THEORY:
    p_NE, p_M, pi_NE, pi_M = spros.get_theory(c_i)

### ПОКА ВСЕ НАПИСАНО ДЛЯ "D"
for env in range(ENV):

    raw_price_history = []
    raw_profit_history = []

    ### Инициализация однородных фирм

    M = Environment["firm_model"]
    firm_params = Environment["firm_params"]

    # firm1 = M(**firm_params)

    # firm2 = M(**firm_params)

    firms = [deepcopy(M(**firm_params)) for i in range(n)]

    mem = []
    
    ### Инициализация памяти платформы
    # -
    
    ### Инициализация памяти в отзывах
    # -

    ### Инициализация основного цикла
    if str(firms[0]) == "epsilon_greedy":
        for t in tqdm(range(T)):

            ### действие
            idx = []
            p = []
            for f in firms:
                idx_i = f.suggest()
                idx.append(idx_i)
                p.append(prices[idx_i])

            ### подсчет спроса
            doli = spros.distribution(p)

            ### подсчет прибыли фирм
            pi = []
            for i in range(n):
                pi_i = (p[i] - c_i) * doli[i]
                pi.append(pi_i)

            raw_profit_history.append(pi)

            ### обновление весов алгоритмов
            for i in range(n):
                f = firms[i]
                f.update(idx[i], pi[i])
                firms[i] = f

            raw_price_history.append(p)
    
    elif str(firms[0]) == "TQL":
        for t in tqdm(range(-MEMORY_VOLUME, T), f"Раунд {env + 1}"):
        # for t in range(-MEMORY_VOLUME, T):

            idx = []
            for i in range(n):
                idx_i = firms[i].suggest()
                idx.append(idx_i)

            learn = mem.copy()
            if t < 0:
                learn.append(idx)
            else:
                learn = learn[1:] + [idx]

            p = []
            for i in range(n):
                p.append(prices[idx[i]])

            doli = spros.distribution(p)

            pi = []
            for i in range(n):
                pi_i = (p[i] - c_i) * doli[i]
                pi.append(pi_i)

            for i in range(n):
                f = firms[i]
                x = learn.copy()
                if len(learn) == MEMORY_VOLUME and not(own) and not(ONLY_OWN):
                    for j in range(MEMORY_VOLUME):
                        x[j] = x[j][: i] + x[j][i + 1 :]
                elif len(learn) == MEMORY_VOLUME and ONLY_OWN:
                    for j in range(MEMORY_VOLUME):
                        x[j] = [x[j][i]]

                f.update(idx[i], x, pi[i])
                firms[i] = f

            for i in range(n):
                f = firms[i]
                x = learn.copy()
                if len(learn) == MEMORY_VOLUME and not(own) and not(ONLY_OWN):
                    for j in range(MEMORY_VOLUME):
                        x[j] = x[j][: i] + x[j][i + 1 :]
                elif len(learn) == MEMORY_VOLUME and ONLY_OWN:
                    for j in range(MEMORY_VOLUME):
                        x[j] = [x[j][i]]
                    
                f.adjust_memory(x)
                firms[i] = f
            
            mem = learn.copy()

            raw_profit_history.append(pi)
            raw_price_history.append(p)
    
    elif str(firms[0]) == "TN_DDQN":
        pass

    raw_price_history = np.array(raw_price_history)
    raw_profit_history = np.array(raw_profit_history)

    Price_history.append(tuple([np.mean(raw_price_history[-int(T/20):, i]) for i in range(n)]))
    Profit_history.append(tuple([np.mean(raw_profit_history[-int(T/20):, i]) for i in range(n)]))


if VISUALIZE or SAVE:

    # plt.figure(figsize=(20, 5))
    # for i in range(n):
    #     plt.plot(raw_price_history[:, i], c = color[i], label = f"Фирма {i + 1}", linewidth= 0.2)

    # plt.show()

    fig, ax = plt.subplots(1, 2 + int(profit_dynamic == "compare"), figsize= (20, 5))

    plotFirst = ax[0]
    plotSecond = ax[1]
    if profit_dynamic == "compare":
        plotThird = ax[2]

    ### Усреднение динамики цены
    window_size = int(0.05*T)
    kernel = np.ones(window_size) / window_size

    for i in range(n):
        mv = np.convolve(raw_price_history[:, i], kernel, mode='valid')\
        
        if profit_dynamic == "real" or profit_dynamic == "compare":
            if i == 0:
                all_mv = mv.copy()
                all_mv = all_mv.reshape(-1, 1)
            else:
                mv = mv.reshape(-1, 1)
                all_mv = np.hstack((all_mv, mv))
        
        plotFirst.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
    
    if VISUALIZE_THEORY:
        plotFirst.plot([p_NE]*len(mv), c = "#6C7B8B", linestyle = "--", label = "NE, M")
        plotFirst.plot([p_M]*len(mv), c = "#6C7B8B", linestyle = "--")
    
    plotFirst.set_title("Динамика цен")
    plotFirst.set_ylabel(f'Сглаженная цена (скользящее среднее по {window_size})')
    plotFirst.set_xlabel('Итерация')
    plotFirst.legend(loc = loc)

    if profit_dynamic == "MA" or profit_dynamic == "compare":
        ### Усреднение динамики прибыли
        window_size = int(0.05*T)
        kernel = np.ones(window_size) / window_size

        for i in range(n):
            mv = np.convolve(raw_profit_history[:, i], kernel, mode='valid')
            plotSecond.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
        
        if VISUALIZE_THEORY:
            plotSecond.plot([pi_NE]*len(mv), c = "#6C7B8B", linestyle = "--", label = "NE, M")
            plotSecond.plot([pi_M]*len(mv), c = "#6C7B8B", linestyle = "--")
        
        plotSecond.set_title("Динамика прибылей")
        plotSecond.set_ylabel(f'Сглаженная прибыль (скользящее среднее по {window_size})')
        plotSecond.set_xlabel('Итерация')
        plotSecond.legend(loc = loc)

    if profit_dynamic == "real" or profit_dynamic == "compare":
        ### Подсчет прибыли для усредненных цен
        a = Environment["demand_params"]["a"]
        mu = Environment["demand_params"]["mu"]

        zeros_column = a * np.ones((all_mv.shape[0], 1), dtype=all_mv.dtype)
        all_d = np.hstack((zeros_column, all_mv))
        all_d = np.exp((a-all_d)/mu)
        s = np.sum(all_d, axis = 1)
        all_d = all_d / s[:, np.newaxis]
        all_d = all_d[:, 1:]
        c = c_i * np.ones(all_mv.shape)
        smoothed_pi = (all_mv - c) * all_d

        for i in range(n):
            mv = smoothed_pi[:, i]
            if profit_dynamic == "compare":
                plotThird.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
            else:
                plotSecond.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
    
    if profit_dynamic != "compare":
        if VISUALIZE_THEORY:
            plotSecond.plot([pi_NE]*len(mv), c = "#6C7B8B", linestyle = "--", label = "NE, M")
            plotSecond.plot([pi_M]*len(mv), c = "#6C7B8B", linestyle = "--")

        plotSecond.set_title("Динамика прибылей")
        plotSecond.set_ylabel(f'Прибыль по сглаженной цене')
        plotSecond.set_xlabel('Итерация')
        plotSecond.legend(loc = loc)
    else:
        if VISUALIZE_THEORY:
            plotThird.plot([pi_NE]*len(mv), c = "#6C7B8B", linestyle = "--", label = "NE, M")
            plotThird.plot([pi_M]*len(mv), c = "#6C7B8B", linestyle = "--")

        plotThird.set_title("Динамика прибылей")
        plotThird.set_ylabel(f'Прибыль по сглаженной цене')
        plotThird.set_xlabel('Итерация')
        plotThird.legend(loc = loc)
    
    plot_name = f'T_{T}_n_{n}_model_{str(firms[0])}_MV_{MEMORY_VOLUME}_own_{own}_mode_{Environment["firm_params"]["mode"]}_profit_dynamic_{profit_dynamic}'

    if SAVE:
        plt.savefig(plot_name, dpi = 1000)

    if VISUALIZE:
        plt.show()


Price_history = np.array(Price_history)
Profit_history = np.array(Profit_history)

print("Средняя цена по всем раундам:", " ".join([str(np.mean(Price_history[:, i])) for i in range(n)]))
print("Средняя прибыль по всем раундам:", " ".join([str(np.mean(Profit_history[:, i])) for i in range(n)]))

"""
Средняя цена по всем раундам: 1.6830796000000001 1.6826728
Средняя прибыль по всем раундам: 0.261182733098004 0.2620053694533933
n = 2, ENV = 100, T = 100000, mode = "zhou", MEMORY_VOLUME = 1, own = False

Средняя цена по всем раундам: 1.7535228999999999 1.7510052999999999 1.7533135
Средняя прибыль по всем раундам: 0.12345687961002637 0.12364908988203568 0.12292997137728587
n = 3, eps = 0.9, ENV = 100, T = 100000, mode = "zhou", MEMORY_VOLUME = 1, own = False
"""