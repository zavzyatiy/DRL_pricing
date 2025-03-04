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
# np.random.seed(42)
# torch.manual_seed(42)

M = Environment["firm_model"]
firm_params = Environment["firm_params"]
TN_DDQN = int(int(str(M(**firm_params)) == "TN_DDQN"))

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
if TN_DDQN == 1:
    arms_amo_price = Environment["arms_amo_price"]
    arms_amo_inv = Environment["arms_amo_inv"]
else:
    arms_amo = Environment["arms_amo_price"]

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
# Выводить итоговую информацию о симуляциях?
SUMMARY = Environment["SUMMARY"]
# С какой стороны отображать легенду на графиках
loc = Environment["loc"]

# Цены
prices = Environment["prices"]
if TN_DDQN == 1:
    inventory = Environment["inventory"]

# Дополнительные параметры, характерные для моделей
if str(M(**firm_params)) == "TQL":
    MEMORY_VOLUME = Environment["firm_params"]["MEMORY_VOLUME"]
    own = Environment["firm_params"]["own"]
    ONLY_OWN = Environment["firm_params"]["ONLY_OWN"]

elif TN_DDQN == 1:
    MEMORY_VOLUME = Environment["firm_params"]["MEMORY_VOLUME"]
    batch_size = Environment["firm_params"]["batch_size"]
    own = Environment["own"]

Price_history = []
Profit_history = []
if TN_DDQN == 1:
    Stock_history = []

demand_params = Environment["demand_params"]
spros = demand_function(**demand_params)
if VISUALIZE_THEORY:
    p_NE, p_M, pi_NE, pi_M = spros.get_theory(c_i)
    inv_NE, inv_M = spros.distribution([p_NE]*n)[0], spros.distribution([p_M]*n)[0]

### ПОКА ВСЕ НАПИСАНО ДЛЯ "D"
for env in range(ENV):

    raw_price_history = []
    raw_profit_history = []
    if TN_DDQN == 1:
        raw_stock_history = []

    ### Инициализация однородных фирм

    firms = [deepcopy(M(**firm_params)) for i in range(n)]

    mem = []

    if TN_DDQN == 1:
        x_t = [0 for i in range(n)]
    
    ### Инициализация платформы
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
        for t in tqdm(range(- MEMORY_VOLUME - batch_size, T), f"Раунд {env + 1}"):
        # for t in range(- MEMORY_VOLUME - batch_size, T):
            # print("!!!!!!!!", t)
            
            idxs = []
            for i in range(n):
                state_i = mem.copy()

                if len(state_i) == MEMORY_VOLUME and not(own):
                    for j in range(MEMORY_VOLUME):
                        state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                
                firm_state = {
                    'current_inventory': x_t[i],
                    'competitors_prices': state_i,
                }

                idxs_i = firms[i].suggest_actions(firm_state)
                idxs.append(idxs_i)

            learn = mem.copy()

            if len(learn) < MEMORY_VOLUME:
                learn.append([x[1] for x in idxs])
            else:
                learn = learn[1:] + [[x[1] for x in idxs]]
            
            inv = [inventory[x[0]] for x in idxs]
            p = [prices[x[1]] for x in idxs]
            # inv = []
            # p = []
            # for i in range(n):
            #     # if inventory[idxs[i][0]] >= x_t[i]:
            #     #     inv.append(inventory[idxs[i][0]])
            #     # else:
            #     #     print("!!!")
            #     #     inv.append(x_t[i])
                
            #     p.append(prices[idxs[i][1]])
            #     inv.append(inventory[idxs[i][0]])

            doli = spros.distribution(p)

            pi = []
            for i in range(n):
                pi_i = p[i] * doli[i] - c_i * (inv[i] - x_t[i]) - h_plus * max(0, inv[i] - doli[i]) - v_minus * min(0, doli[i] - inv[i])
                pi.append(pi_i)

            if len(learn) == MEMORY_VOLUME:
                for i in range(n):
                    state_i = mem.copy()
                    if len(state_i) == MEMORY_VOLUME and not(own):
                        for j in range(MEMORY_VOLUME):
                            state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                    
                    new = learn.copy()
                    if len(new) == MEMORY_VOLUME and not(own):
                        for j in range(MEMORY_VOLUME):
                            new[j] = new[j][: i] + new[j][i + 1 :]
                    
                    prev_state = {
                        'current_inventory': x_t[i],
                        'competitors_prices': state_i,
                    }

                    new_state = {
                        'current_inventory': max(0, inv[i] - doli[i]),
                        'competitors_prices': new,
                    }

                    firms[i].cache_experience(prev_state, idxs[i], pi[i], new_state)

            for i in range(n):
                x_t[i] = max(0, inv[i] - doli[i])

            for i in range(n):
                firms[i].update()
            
            mem = learn.copy()

            raw_profit_history.append(pi)
            raw_price_history.append(p)
            raw_stock_history.append(inv)

    raw_price_history = np.array(raw_price_history)
    raw_profit_history = np.array(raw_profit_history)
    if TN_DDQN == 1:
        raw_stock_history = np.array(raw_stock_history)

    Price_history.append(tuple([np.mean(raw_price_history[-int(T/20):, i]) for i in range(n)]))
    Profit_history.append(tuple([np.mean(raw_profit_history[-int(T/20):, i]) for i in range(n)]))
    if TN_DDQN == 1:
        Stock_history.append(tuple([np.mean(raw_stock_history[-int(T/20):, i]) for i in range(n)]))


if VISUALIZE or SAVE:

    profit_dynamic = TN_DDQN * "MA" + (1 - TN_DDQN) * profit_dynamic
    fig, ax = plt.subplots(1 + TN_DDQN, 2 + (1 - TN_DDQN) * int(profit_dynamic == "compare"), figsize= (20, 5*(1 + TN_DDQN*1.1)))

    if TN_DDQN == 0:
        plotFirst = ax[0]
        plotSecond = ax[1]
        if profit_dynamic == "compare":
            plotThird = ax[2]
    else:
        plotFirst = ax[0][0]
        plotSecond = ax[0][1]
        plotStock = ax[1][0]
        if profit_dynamic == "compare":
            plotThird = ax[1][1]

    ### Усреднение динамики цены
    window_size = int(0.05*T)
    kernel = np.ones(window_size) / window_size

    for i in range(n):
        mv = np.convolve(raw_price_history[:, i], kernel, mode='valid')
        
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

    if TN_DDQN == 1:
        ### Усреднение динамики запасов
        window_size = int(0.05*T)
        kernel = np.ones(window_size) / window_size

        for i in range(n):
            mv = np.convolve(raw_stock_history[:, i], kernel, mode='valid')
            
            if profit_dynamic == "real" or profit_dynamic == "compare":
                if i == 0:
                    all_mv = mv.copy()
                    all_mv = all_mv.reshape(-1, 1)
                else:
                    mv = mv.reshape(-1, 1)
                    all_mv = np.hstack((all_mv, mv))
            
            plotStock.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
        
        
        if VISUALIZE_THEORY:
            plotStock.plot([inv_NE]*len(mv), c = "#6C7B8B", linestyle = "--", label = "M, NE")
            plotStock.plot([inv_M]*len(mv), c = "#6C7B8B", linestyle = "--")
        
        plotStock.set_title("Динамика запасов")
        plotStock.set_ylabel(f'Сглаженные объемы (скользящее среднее по {window_size})')
        plotStock.set_xlabel('Итерация')
        plotStock.legend(loc = loc)

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


if SUMMARY:
    Price_history = np.array(Price_history)
    Profit_history = np.array(Profit_history)
    # print(Price_history)
    # print(Profit_history)

    print(f"Средняя цена по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Price_history[:, i]), 3)) for i in range(n)]))

    if TN_DDQN == 1:
        Stock_history = np.array(Stock_history)
        # print(Stock_history)
        print(f"Среднии запасы по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Stock_history[:, i]), 3)) for i in range(n)]))

    print(f"Средняя прибыль по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Profit_history[:, i]), 3)) for i in range(n)]))

    print("-"*20*n)
    print("Теоретические цены:", round(p_NE , 3), round(p_M , 3))

    if TN_DDQN == 1:
        print("Теоретические инв. в запасы:", round(inv_NE , 3), round(inv_M , 3))

    print("Теоретические прибыли:", round(pi_NE , 3), round(pi_M , 3))

    print("-"*20*n)
    print("Индекс сговора по цене:", str(round(100 * (np.mean(Price_history) - p_NE)/(p_M - p_NE), 2)) + "%")

    if TN_DDQN == 1:
        print("Индекс сговора по запасам:", str(round(100 * (np.mean(Stock_history) - inv_NE)/(inv_M - inv_NE), 2)) + "%")

    print("Индекс сговора по прибыли:", str(round(100 * (np.mean(Profit_history) - pi_NE)/(pi_M - pi_NE), 2)) + "%")


"""
Средняя цена по всем раундам: 1.6830796000000001 1.6826728
Средняя прибыль по всем раундам: 0.261182733098004 0.2620053694533933
n = 2, ENV = 100, T = 100000, mode = "zhou", MEMORY_VOLUME = 1, own = False

Средняя цена по всем раундам: 1.7535228999999999 1.7510052999999999 1.7533135
Средняя прибыль по всем раундам: 0.12345687961002637 0.12364908988203568 0.12292997137728587
n = 3, eps = 0.9, ENV = 100, T = 100000, mode = "zhou", MEMORY_VOLUME = 1, own = False

T = 10000, ENV = 30, h_plus = 1.17498, v_minus = 1.17498, n = 2, arms = 21 (оба), batch_size = 128
Средняя цена по последним 500 раундов: 0.858 0.837
Средняя прибыль по последним 500 раундов: 0.24 0.265
Среднии запасы по последним 500 раундов: 0.467 0.402
----------------------------------------
Теоретические цены: 0.723 1.175
Теоретические прибыли: 0.223 0.337
Теоретические инв. в запасы: 0.471 0.365
----------------------------------------
Индекс сговора по цене: 27.59%
Индекс сговора по прибыли: 25.81%
Индекс сговора по запасам: 34.82%
"""