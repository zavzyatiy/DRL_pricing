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

    firm1 = M(**firm_params)

    firm2 = M(**firm_params)

    mem1 = []
    mem2 = []
    
    ### Инициализация памяти платформы
    # -
    
    ### Инициализация памяти в отзывах
    # -

    ### Инициализация основного цикла
    if str(firm1) == "epsilon_greedy" and str(firm2) == "epsilon_greedy":
        for t in tqdm(range(T)):

            ### действие
            idx1 = firm1.suggest()
            idx2 = firm2.suggest()
            p1 = prices[idx1]
            p2 = prices[idx2]

            ### подсчет спроса
            doli = spros.distribution([p1, p2])

            ### подсчет прибыли фирм
            pi_1 = (p1 - c_i) * doli[0]
            pi_2 = (p2 - c_i) * doli[1]

            raw_profit_history.append((pi_1, pi_2))

            ### обновление весов алгоритмов
            firm1.update(idx1, pi_1)
            firm2.update(idx2, pi_2)

            raw_price_history.append((p1, p2))
    
    elif str(firm1) == "TQL" and str(firm2) == "TQL":
        for t in tqdm(range(T + MEMORY_VOLUME), f"Раунд {env + 1}"):
        # for t in range(T + MEMORY_VOLUME):

            idx1 = firm1.suggest(mem1)
            idx2 = firm2.suggest(mem2)

            if firm1.t <= 0:
                mem1.append(idx2)
                mem2.append(idx1)
            else:
                mem1 = mem1[1:] + [idx2]
                mem2 = mem2[1:] + [idx1]

            p1 = prices[idx1]
            p2 = prices[idx2]

            doli = spros.distribution([p1, p2])

            pi_1 = (p1 - c_i) * doli[0]
            pi_2 = (p2 - c_i) * doli[1]

            firm1.update(idx1, mem1, pi_1)
            firm2.update(idx2, mem2, pi_2)

            raw_profit_history.append((pi_1, pi_2))
            raw_price_history.append((p1, p2))
    
    raw_price_history = np.array(raw_price_history)
    raw_profit_history = np.array(raw_profit_history)

    Price_history.append((np.mean(raw_price_history[:, 0]), np.mean(raw_price_history[:, 1])))
    Profit_history.append((np.mean(raw_profit_history[:, 0]), np.mean(raw_profit_history[:, 1])))


if VISUALIZE or SAVE:

    fig, ax = plt.subplots(1, 2, figsize= (15, 6))

    plotFirst = ax[0]
    plotSecond = ax[1]

    window_size = int(T/20)
    kernel = np.ones(window_size) / window_size
    mv1 = np.convolve(raw_price_history[:, 0], kernel, mode='valid')
    mv2 = np.convolve(raw_price_history[:, 1], kernel, mode='valid')

    plotFirst.plot(mv1) # , linewidth= 0.2)
    plotFirst.plot(mv2) # , linewidth= 0.2)
    plotFirst.set_title("Динамика цен")
    plotFirst.set_ylabel(f'Сглаженная цена (скользящее среднее по {window_size})')
    plotFirst.set_xlabel('Итерация')


    window_size = int(0.05*T)
    kernel = np.ones(window_size) / window_size
    mv1 = np.convolve(raw_profit_history[:, 0], kernel, mode='valid')
    mv2 = np.convolve(raw_profit_history[:, 1], kernel, mode='valid')

    plotSecond.plot(mv1) # , linewidth= 0.2)
    plotSecond.plot(mv2) # , linewidth= 0.2)
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