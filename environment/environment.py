### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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
# Выводить итоговый график?
VISUALIZE = Environment["VISUALIZE"]
# Сохранить итоговоый график?
SAVE = Environment["SAVE"]

# Цены
prices = Environment["prices"]
MEMORY_VOLUME = Environment["firm_params"]["MEMORY_VOLUME"]

Price_history = []
Profit_history = []

### ПОКА ВСЕ НАПИСАНО ДЛЯ "D"
for env in range(ENV):

    raw_price_history = []
    raw_profit_history = []

    demand_params = Environment["demand_params"]
    spros = demand_function(**demand_params)

    ### Инициализация однородных фирм

    M = Environment["firm_model"]
    firm_params = Environment["firm_params"]

    # firm1 = M(**firm_params)

    # firm2 = M(**firm_params)

    firms = [M(**firm_params) for i in range(n)]

    mem = [[] for i in range(n)]
    
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
        # for t in range(T + MEMORY_VOLUME):
            idx = []
            for i in range(n):
                idx_i = firms[i].suggest(mem[i])
                idx.append(idx_i)

            if t < 0:
                for i in range(n):
                    mem[i].append(idx[i])
            else:
                for i in range(n):
                    x = mem[i][1:]
                    mem[i] = x + [idx[i]]
            
            # print(mem)

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
                f.update(idx[i], mem[i], pi[i])
                firms[i] = f

            raw_profit_history.append(pi)
            raw_price_history.append(p)
    
    raw_price_history = np.array(raw_price_history)
    raw_profit_history = np.array(raw_profit_history)

    Price_history.append((np.mean(raw_price_history[:, 0]), np.mean(raw_price_history[:, 1])))
    Profit_history.append((np.mean(raw_profit_history[:, 0]), np.mean(raw_profit_history[:, 1])))


if VISUALIZE or SAVE:

    fig, ax = plt.subplots(1, 2, figsize= (15, 6))

    plotFirst = ax[0]
    plotSecond = ax[1]

    window_size = int(0.05*T)
    kernel = np.ones(window_size) / window_size

    for i in range(n):
        mv = np.convolve(raw_price_history[:, i], kernel, mode='valid') 
        plotFirst.plot(mv, c = color[i]) # , linewidth= 0.2)
    
    plotFirst.set_title("Динамика цен")
    plotFirst.set_ylabel(f'Сглаженная цена (скользящее среднее по {window_size})')
    plotFirst.set_xlabel('Итерация')

    window_size = int(0.05*T)
    kernel = np.ones(window_size) / window_size

    for i in range(n):
        mv = np.convolve(raw_profit_history[:, i], kernel, mode='valid') 
        plotSecond.plot(mv, c = color[i]) # , linewidth= 0.2)
    
    plotSecond.set_title("Динамика прибылей")
    plotSecond.set_ylabel(f'Сглаженная прибыль (скользящее среднее по {window_size})')
    plotSecond.set_xlabel('Итерация')

    plot_name = ""

    if SAVE:
        plt.savefig(".png", dpi = 1000)

    if VISUALIZE:
        plt.show()

Price_history = np.array(Price_history)
Profit_history = np.array(Profit_history)
print("Средняя цена по всем раундам:", np.mean(Price_history[:, 0]), np.mean(Price_history[:, 1]))
print("Средняя прибыль по всем раундам:", np.mean(Profit_history[:, 0]), np.mean(Profit_history[:, 1]))

"""
Средняя цена по всем раундам: 1.63549825501745 1.6407975932740673
Средняя прибыль по всем раундам: 0.27545766483096434 0.2725363997035166
ENV = 100, T = 200000, mode = "zhou"
"""