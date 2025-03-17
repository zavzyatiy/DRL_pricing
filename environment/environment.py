### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt

import random
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import os
import json

# from firms_RL import epsilon_greedy, TQL
from specification import Environment, demand_function

### иницализация randomseed
RANDOM_SEED = Environment["RANDOM_SEED"]
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Инициализация типа модели
M = Environment["firm_model"]
firm_params = Environment["firm_params"]
# Используем ли мы постановку задачи с инвестициями в запасы?
# Модели сопряжены с ней автоматически, так как в работе
# других ситуаций не рассматривается
HAS_INV = int(str(M(**firm_params)) in ["TN_DDQN", "PPO_D", "PPO_C", "SAC"])

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
if HAS_INV == 1:
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
# Сохранять ли итоговую информацию?
SAVE_SUMMARY = Environment["SAVE_SUMMARY"]
SUMMARY = SUMMARY or SAVE_SUMMARY
# Выводить ли итоги записей каждой среды?
SHOW_PROM_RES = Environment["SHOW_PROM_RES"]
# С какой стороны отображать легенду на графиках
loc = Environment["loc"]

# Цены
prices = Environment["prices"]
if HAS_INV == 1:
    inventory = Environment["inventory"]

# Дополнительные параметры, характерные для моделей
if str(M(**firm_params)) == "TQL":
    MEMORY_VOLUME = Environment["firm_params"]["MEMORY_VOLUME"]
    own = Environment["firm_params"]["own"]
    ONLY_OWN = Environment["firm_params"]["ONLY_OWN"]
elif str(M(**firm_params)) == "TN_DDQN":
    MEMORY_VOLUME = Environment["firm_params"]["MEMORY_VOLUME"]
    batch_size = Environment["firm_params"]["batch_size"]
    own = Environment["own"]
elif str(M(**firm_params)) == "PPO_D":
    MEMORY_VOLUME = Environment["MEMORY_VOLUME"]
    own = Environment["own"]
    batch_size = Environment["firm_params"]["batch_size"]
    N_epochs = Environment["firm_params"]["N_epochs"]
    epochs = Environment["firm_params"]["epochs"]
    assert (N_epochs >= batch_size) # and (N_epochs >= epochs)
elif str(M(**firm_params)) == "PPO_C":
    MEMORY_VOLUME = Environment["MEMORY_VOLUME"]
    own = Environment["own"]
    batch_size = Environment["firm_params"]["batch_size"]
    N_epochs = Environment["firm_params"]["N_epochs"]
    epochs = Environment["firm_params"]["epochs"]
    assert (N_epochs >= batch_size) # and (N_epochs >= epochs)

Price_history = []
Profit_history = []
if HAS_INV == 1:
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
    if HAS_INV == 1:
        raw_stock_history = []

    ### Инициализация однородных фирм

    firms = [deepcopy(M(**firm_params)) for i in range(n)]

    mem = []

    if HAS_INV == 1:
        x_t = np.array([0 for i in range(n)])
    
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
            
            inv = np.array([inventory[x[0]] for x in idxs])
            p = np.array([prices[x[1]] for x in idxs])

            doli = spros.distribution(p)

            # pi = []
            # pi_inv = []
            # pi_price = []
            # for i in range(n):
                # pi_i = p[i] * doli[i] - c_i * (inv[i] - x_t[i]) - h_plus * max(0, inv[i] - doli[i]) - v_minus * min(0, doli[i] - inv[i])
                # pi.append(pi_i)
                # pi_inv_i = - c_i * (inv[i] - x_t[i]) - h_plus * max(0, inv[i] - doli[i]) - v_minus * min(0, doli[i] - inv[i])
                # pi_price_i = p[i] * doli[i]
                # pi_inv.append(pi_inv_i)
                # pi_price.append(pi_price_i)
            # print(inv - doli)

            pi = p * doli - c_i * (inv - np.array(x_t)) - h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)

            # print("Цены", p)
            # print("Инв", inv)
            # print("Было", x_t)
            # print("Доли", doli)
            # print("-"*100)
            # print(p * doli)
            # print(- c_i * (inv - np.array(x_t)))
            # print("Вообще", inv - doli)
            # print(- h_plus * np.maximum(0, inv - doli))
            # print(v_minus * np.minimum(0, inv - doli))
            # print("#"*100)

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
                    # firms[i].cache_experience(prev_state, idxs[i], (pi_inv[i], pi_price[i]), new_state)

            # for i in range(n):
            #     x_t[i] = max(0, inv[i] - doli[i])
            x_t = np.maximum(0, inv - doli)

            for i in range(n):
                firms[i].update()
            
            mem = learn.copy()

            raw_profit_history.append(pi)
            raw_price_history.append(p)
            raw_stock_history.append(inv)

    elif str(firms[0]) == "PPO_D":
        total_t = -MEMORY_VOLUME
        with tqdm(total = T + MEMORY_VOLUME, desc=f'Раунд {env + 1}') as pbar:
            while total_t < T:
                # for t in tqdm(range(- MEMORY_VOLUME - batch_size, T), f"Раунд {env + 1}"):
                if total_t < 0:
                    min_t = total_t
                    max_t = 0
                else:
                    min_t = total_t
                    max_t = min(total_t + N_epochs, T)
                
                for t in range(min_t, max_t):

                    # print("!!!", t)

                    idxs = []
                    # print("ЗАПАСЫ", x_t)
                    for i in range(n):
                        # print("Фирма:", i)
                        state_i = mem.copy()

                        if len(state_i) == MEMORY_VOLUME and not(own):
                            for j in range(MEMORY_VOLUME):
                                state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                        
                        firm_state = {
                            'current_inventory': x_t[i],
                            'competitors_prices': state_i,
                        }

                        if t >= 0:
                            idxs_i = firms[i].suggest_actions(firm_state)
                        else:
                            idxs_i = (random.sample([i for i in range(len(inventory))], 1)[0],
                                      random.sample([i for i in range(len(prices))], 1)[0])
                        # print("Действия", idxs_i)
                        idxs.append(idxs_i)

                    learn = mem.copy()

                    if len(learn) < MEMORY_VOLUME:
                        learn.append([x[1] for x in idxs])
                    else:
                        learn = learn[1:] + [[x[1] for x in idxs]]
                    
                    inv = np.array([inventory[x[0]] for x in idxs])
                    p = np.array([prices[x[1]] for x in idxs])

                    doli = spros.distribution(p)

                    pi = p * doli - c_i * (inv - np.array(x_t)) - h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                    # print("-"*50)
                    # print("Итерация", t, total_t)
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
                            # print(f"Память фирмы {i}", firms[i].memory)
                            
                    # for i in range(n):
                    #     x_t[i] = max(0, inv[i] - doli[i])
                    x_t = np.maximum(0, inv - doli)
                    
                    mem = learn.copy()

                    pbar.update(1)
                    raw_profit_history.append(pi)
                    raw_price_history.append(p)
                    raw_stock_history.append(inv)
                
                if total_t < 0:
                    total_t = 0
                else:
                    # print("#"*50)
                    for i in range(n):
                        # print("Обновление фирмы", i)
                        firms[i].update()
                    
                    total_t = min(total_t + N_epochs, T)
                
    elif str(firms[0]) == "PPO_C":
        total_t = -MEMORY_VOLUME
        with tqdm(total = T + MEMORY_VOLUME, desc=f'Раунд {env + 1}') as pbar:
            while total_t < T:
                # for t in tqdm(range(- MEMORY_VOLUME - batch_size, T), f"Раунд {env + 1}"):
                if total_t < 0:
                    min_t = total_t
                    max_t = 0
                else:
                    min_t = total_t
                    max_t = min(total_t + N_epochs, T)
                
                for t in range(min_t, max_t):

                    # print("!!!", t)

                    acts = []
                    iter_probs = []
                    # print("ЗАПАСЫ", x_t)
                    for i in range(n):
                        # print("Фирма:", i)
                        state_i = mem.copy()

                        if len(state_i) == MEMORY_VOLUME and not(own):
                            for j in range(MEMORY_VOLUME):
                                state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                        
                        firm_state = {
                            'current_inventory': x_t[i],
                            'competitors_prices': state_i,
                        }

                        if t >= 0:
                            inv, price, u_inv, u_prc = firms[i].suggest_actions(firm_state)
                            acts_i = (inv, price)
                        else:
                            u_inv = torch.distributions.Normal(0, 1).sample()
                            u_prc = torch.distributions.Normal(0, 1).sample()
                            act_inv = x_t[i] + torch.sigmoid(u_inv/10) * (inventory[1] - x_t[i])
                            act_price = prices[0] + torch.sigmoid(u_prc/10) * (prices[1] - prices[0])
                            acts_i = (act_inv, act_price)
                        # print("Фирма", i)
                        # print("iter_probs", (u_inv, u_prc))
                        iter_probs.append((u_inv, u_prc))
                        acts.append(acts_i)

                    learn = mem.copy()

                    if len(learn) < MEMORY_VOLUME:
                        learn.append([x[1] for x in acts])
                    else:
                        learn = learn[1:] + [[x[1] for x in acts]]
                    
                    inv = np.array([x[0] for x in acts])
                    p = np.array([x[1] for x in acts])
                    # print(p)

                    doli = spros.distribution(p)

                    pi = p * doli - c_i * (inv - np.array(x_t)) - h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                    # print("-"*50)
                    # print("Итерация", t, total_t)
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

                            firms[i].cache_experience(prev_state, iter_probs[i], pi[i], new_state)
                            # print(f"Память фирмы {i}", firms[i].memory)
                            
                    # for i in range(n):
                    #     x_t[i] = max(0, inv[i] - doli[i])
                    x_t = np.maximum(0, inv - doli)
                    
                    mem = learn.copy()

                    pbar.update(1)
                    raw_profit_history.append(pi)
                    raw_price_history.append(p)
                    raw_stock_history.append(inv)
                
                if total_t < 0:
                    total_t = 0
                else:
                    # print("#"*50)
                    for i in range(n):
                        # print("Обновление фирмы", i)
                        firms[i].update()
                    
                    total_t = min(total_t + N_epochs, T)

    elif str(firms[0]) == "SAC":

        continue

    raw_price_history = np.array(raw_price_history)
    raw_profit_history = np.array(raw_profit_history)
    if HAS_INV == 1:
        raw_stock_history = np.array(raw_stock_history)

    Price_history.append(tuple([np.mean(raw_price_history[-int(T/20):, i]) for i in range(n)]))
    Profit_history.append(tuple([np.mean(raw_profit_history[-int(T/20):, i]) for i in range(n)]))
    if SHOW_PROM_RES:
        print("\n", Price_history[-1])
    if HAS_INV == 1:
        Stock_history.append(tuple([np.mean(raw_stock_history[-int(T/20):, i]) for i in range(n)]))
        print(Stock_history[-1])
    if SHOW_PROM_RES:
        print(Profit_history[-1])
        print("-"*100)
        print("\n")


if VISUALIZE or SAVE:

    sgladit = int(0.05 * T)
    fig, ax = plt.subplots(1 + HAS_INV, 2 + (1 - HAS_INV) * int(profit_dynamic == "compare"), figsize= (20, 5*(1 + HAS_INV*1.1)))

    if HAS_INV == 0:
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
    window_size = sgladit
    kernel = np.ones(window_size) / window_size

    for i in range(n):
        mv = np.convolve(raw_price_history[:, i], kernel, mode='valid')
        # print("ПРОВЕРКА НА АДЕКВАТНОСТЬ")
        # print(raw_price_history[:, i], mv)
        
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

    if HAS_INV == 1:
        ### Усреднение динамики запасов
        window_size = sgladit
        kernel = np.ones(window_size) / window_size

        for i in range(n):
            mv = np.convolve(raw_stock_history[:, i], kernel, mode='valid')
            
            if profit_dynamic == "real" or profit_dynamic == "compare":
                if i == 0:
                    all_inv = mv.copy()
                    all_inv = all_inv.reshape(-1, 1)
                else:
                    mv = mv.reshape(-1, 1)
                    all_inv = np.hstack((all_inv, mv))
            
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
        window_size = sgladit
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

        if HAS_INV == 0:
            zeros_column = a * np.ones((all_mv.shape[0], 1), dtype=all_mv.dtype)
            all_d = np.hstack((zeros_column, all_mv))
            all_d = np.exp((a-all_d)/mu)
            s = np.sum(all_d, axis = 1)
            all_d = all_d / s[:, np.newaxis]
            all_d = all_d[:, 1:]
            c = c_i * np.ones(all_mv.shape)
            smoothed_pi = (all_mv - c) * all_d
        else:
            zeros_column = a * np.ones((all_mv.shape[0], 1), dtype=all_mv.dtype)
            all_d = np.hstack((zeros_column, all_mv))
            all_d = np.exp((a-all_d)/mu)
            s = np.sum(all_d, axis = 1)
            all_d = all_d / s[:, np.newaxis]
            all_d = all_d[:, 1:]
            stocks = np.concatenate((np.array([[0, 0]]), np.maximum(0, all_inv - all_d)))[:-1]
            smoothed_pi = all_mv * all_d - c_i * (all_inv - stocks) - h_plus * np.maximum(0, all_inv - all_d) + v_minus * np.minimum(0, all_inv - all_d)
            # print("-"*100)
            # print(all_mv * all_d)
            # print(- c_i * (all_inv - stocks))
            # print(- h_plus * np.maximum(0, all_inv - all_d))
            # print(v_minus * np.minimum(0, all_inv - all_d))

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
            plotThird.set_ylabel(f'Прибыль по сглаженной цене' + HAS_INV*" и запасам")
            plotThird.set_xlabel('Итерация')
            plotThird.legend(loc = loc)


if SUMMARY:
    print("\n")
    Price_history = np.array(Price_history)
    Profit_history = np.array(Profit_history)

    print(f"Средняя цена по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Price_history[:, i]), 3)) for i in range(n)]))

    if HAS_INV == 1:
        Stock_history = np.array(Stock_history)
        print(f"Среднии запасы по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Stock_history[:, i]), 3)) for i in range(n)]))

    print(f"Средняя прибыль по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Profit_history[:, i]), 3)) for i in range(n)]))

    print("-"*20*n)
    print("Теоретические цены:", round(p_NE , 3), round(p_M , 3))

    if HAS_INV == 1:
        print("Теоретические инв. в запасы:", round(inv_NE , 3), round(inv_M , 3))

    print("Теоретические прибыли:", round(pi_NE , 3), round(pi_M , 3))

    print("-"*20*n)
    print("Индекс сговора по цене:", str(round(100 * (np.mean(Price_history) - p_NE)/(p_M - p_NE), 2)) + "%")

    if HAS_INV == 1:
        print("Индекс сговора по запасам:", str(round(100 * (np.mean(Stock_history) - inv_NE)/(inv_M - inv_NE), 2)) + "%")

    print("Индекс сговора по прибыли:", str(round(100 * (np.mean(Profit_history) - pi_NE)/(pi_M - pi_NE), 2)) + "%")


def convert_ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if SAVE_SUMMARY or VISUALIZE:
    if VISUALIZE and not(SAVE_SUMMARY):

        plot_name = f'T_{T}_n_{n}_model_{str(firms[0])}_MV_{MEMORY_VOLUME}_own_{own}_profit_dynamic_{profit_dynamic}'

        if str(firms[0]) == "TN_DDQN":
            plot_name = plot_name + f'_mode_{Environment["firm_params"]["mode"]}'

        if SAVE:
            plt.savefig(plot_name, dpi = 1000)
    
    if SAVE_SUMMARY:
        folders = []
        for f in os.listdir("./DRL_pricing/environment/simulation_results/"):
            if not("." in str(f)) and (str(firms[0]) in str(f)):
                folders.append(f)
        
        res_name = f"./DRL_pricing/environment/simulation_results/{str(firms[0])}_{len(folders) + 1}/"
        
        if not os.path.exists(res_name):
            os.makedirs(res_name)

        with open(res_name + "params.txt", "w+", encoding="utf-8") as f:
            to_write = deepcopy(Environment)
            to_write["firm_model"] = str(firms[0])
            to_write = convert_ndarray_to_list(to_write)
            json.dump(to_write, f, indent=4)
        
        path = os.path.join(res_name, "Price_history.npy")
        np.save(path, Price_history)
        path = os.path.join(res_name, "Profit_history.npy")
        np.save(path, Profit_history)
        if HAS_INV == 1:
            path = os.path.join(res_name, "Stock_history.npy")
            np.save(path, Stock_history)

        with open(res_name + "summary.txt", "w+", encoding="utf-8") as f:
            A = ""
            A = A + f"Средняя цена по последним {int(T/20)} раундов: " + " ".join([str(round(np.mean(Price_history[:, i]), 3)) for i in range(n)])
            A = A + "\n"
            if HAS_INV == 1:
                A = A + f"Среднии запасы по последним {int(T/20)} раундов: " + " ".join([str(round(np.mean(Stock_history[:, i]), 3)) for i in range(n)])
                A = A + "\n"

            A = A + f"Средняя прибыль по последним {int(T/20)} раундов: " + " ".join([str(round(np.mean(Profit_history[:, i]), 3)) for i in range(n)])
            A = A + "\n"

            A = A + "-"*20*n
            A = A + "\n"
            A = A + "Теоретические цены: " + f"{round(p_NE , 3)}, {round(p_M , 3)}"
            A = A + "\n"

            if HAS_INV == 1:
                A = A + "Теоретические инв. в запасы: " + f"{round(inv_NE , 3)}, {round(inv_M , 3)}"
                A = A + "\n"

            A = A + "Теоретические прибыли: " + f"{round(pi_NE , 3)}, {round(pi_M , 3)}"
            A = A + "\n"

            A = A + "-"*20*n
            A = A + "\n"
            A = A + "Индекс сговора по цене: " + str(round(100 * (np.mean(Price_history) - p_NE)/(p_M - p_NE), 2)) + "%"
            A = A + "\n"

            if HAS_INV == 1:
                A = A + "Индекс сговора по запасам: " + str(round(100 * (np.mean(Stock_history) - inv_NE)/(inv_M - inv_NE), 2)) + "%"
                A = A + "\n"

            A = A + "Индекс сговора по прибыли: " + str(round(100 * (np.mean(Profit_history) - pi_NE)/(pi_M - pi_NE), 2)) + "%"
            A = A + "\n"
            f.write(A)

        if VISUALIZE and SAVE_SUMMARY:
            if SAVE:
                plt.savefig(res_name + "plot_last_res", dpi = 1000)


if VISUALIZE:
    plt.show()


"""
Средняя цена по последним 5000 раундов: 1.513 1.491
Среднии запасы по последним 5000 раундов: 13.248 14.401
Средняя прибыль по последним 5000 раундов: 4.625 4.818
----------------------------------------
Теоретические цены: 1.473 1.925
Теоретические инв. в запасы: 14.141 10.946
Теоретические прибыли: 6.687 10.125
----------------------------------------
Индекс сговора по цене: 6.44%
Индекс сговора по запасам: 9.93%
Индекс сговора по прибыли: -57.19%

{'T': 100000, 'ENV': 100, 'n': 2, 'm': 5, 'delta': 0.95, 'gamma': 0.5, 'c_i': 1, 'h_plus': 3, 'v_minus': 3, 'eta': 0.05, 'color': ['#FF7F00', '#1874CD', '#548B54', '#CD2626', '#CDCD00'], 'profit_dynamic': 'compare', 'loc': 'lower left', 'VISUALIZE_THEORY': True, 'VISUALIZE': True, 'SAVE': False, 'SUMMARY': True, 'p_inf': 1, 'p_sup': 2.5, 'arms_amo_price': 101, 'arms_amo_inv': 101, 'demand_params': {'n': 2, 'mode': 'logit', 'a': 2, 'mu': 0.25, 'C': 
30}, 'prices': array([1.   , 1.015, 1.03 , 1.045, 1.06 , 1.075, 1.09 , 1.105, 1.12 ,
       1.135, 1.15 , 1.165, 1.18 , 1.195, 1.21 , 1.225, 1.24 , 1.255,
       1.27 , 1.285, 1.3  , 1.315, 1.33 , 1.345, 1.36 , 1.375, 1.39 ,
       1.405, 1.42 , 1.435, 1.45 , 1.465, 1.48 , 1.495, 1.51 , 1.525,
       1.54 , 1.555, 1.57 , 1.585, 1.6  , 1.615, 1.63 , 1.645, 1.66 ,
       1.675, 1.69 , 1.705, 1.72 , 1.735, 1.75 , 1.765, 1.78 , 1.795,
       1.81 , 1.825, 1.84 , 1.855, 1.87 , 1.885, 1.9  , 1.915, 1.93 ,
       1.945, 1.96 , 1.975, 1.99 , 2.005, 2.02 , 2.035, 2.05 , 2.065,
       2.08 , 2.095, 2.11 , 2.125, 2.14 , 2.155, 2.17 , 2.185, 2.2  ,
       2.215, 2.23 , 2.245, 2.26 , 2.275, 2.29 , 2.305, 2.32 , 2.335,
       2.35 , 2.365, 2.38 , 2.395, 2.41 , 2.425, 2.44 , 2.455, 2.47 ,
       2.485, 2.5  ]), 'inventory': array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8,  2.1,  2.4,  2.7,  3. ,
        3.3,  3.6,  3.9,  4.2,  4.5,  4.8,  5.1,  5.4,  5.7,  6. ,  6.3,
        6.6,  6.9,  7.2,  7.5,  7.8,  8.1,  8.4,  8.7,  9. ,  9.3,  9.6,
        9.9, 10.2, 10.5, 10.8, 11.1, 11.4, 11.7, 12. , 12.3, 12.6, 12.9,
       13.2, 13.5, 13.8, 14.1, 14.4, 14.7, 15. , 15.3, 15.6, 15.9, 16.2,
       16.5, 16.8, 17.1, 17.4, 17.7, 18. , 18.3, 18.6, 18.9, 19.2, 19.5,
       19.8, 20.1, 20.4, 20.7, 21. , 21.3, 21.6, 21.9, 22.2, 22.5, 22.8,
       23.1, 23.4, 23.7, 24. , 24.3, 24.6, 24.9, 25.2, 25.5, 25.8, 26.1,
       26.4, 26.7, 27. , 27.3, 27.6, 27.9, 28.2, 28.5, 28.8, 29.1, 29.4,
       29.7, 30. ]), 'firm_model': <class 'firms_RL.TN_DDQN>, 'firm_params': {'state_dim': 2, 'inventory_actions': array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8,  2.1,  2.4,  2.7,  3. ,
        3.3,  3.6,  3.9,  4.2,  4.5,  4.8,  5.1,  5.4,  5.7,  6. ,  6.3,
        6.6,  6.9,  7.2,  7.5,  7.8,  8.1,  8.4,  8.7,  9. ,  9.3,  9.6,
        9.9, 10.2, 10.5, 10.8, 11.1, 11.4, 11.7, 12. , 12.3, 12.6, 12.9,
       13.2, 13.5, 13.8, 14.1, 14.4, 14.7, 15. , 15.3, 15.6, 15.9, 16.2,
       16.5, 16.8, 17.1, 17.4, 17.7, 18. , 18.3, 18.6, 18.9, 19.2, 19.5,
       19.8, 20.1, 20.4, 20.7, 21. , 21.3, 21.6, 21.9, 22.2, 22.5, 22.8,
       23.1, 23.4, 23.7, 24. , 24.3, 24.6, 24.9, 25.2, 25.5, 25.8, 26.1,
       26.4, 26.7, 27. , 27.3, 27.6, 27.9, 28.2, 28.5, 28.8, 29.1, 29.4,
       29.7, 30. ]), 'price_actions': array([1.   , 1.015, 1.03 , 1.045, 1.06 , 1.075, 1.09 , 1.105, 1.12 ,
       1.135, 1.15 , 1.165, 1.18 , 1.195, 1.21 , 1.225, 1.24 , 1.255,
       1.27 , 1.285, 1.3  , 1.315, 1.33 , 1.345, 1.36 , 1.375, 1.39 ,
       1.405, 1.42 , 1.435, 1.45 , 1.465, 1.48 , 1.495, 1.51 , 1.525,
       1.54 , 1.555, 1.57 , 1.585, 1.6  , 1.615, 1.63 , 1.645, 1.66 ,
       1.675, 1.69 , 1.705, 1.72 , 1.735, 1.75 , 1.765, 1.78 , 1.795,
       1.81 , 1.825, 1.84 , 1.855, 1.87 , 1.885, 1.9  , 1.915, 1.93 ,
       1.945, 1.96 , 1.975, 1.99 , 2.005, 2.02 , 2.035, 2.05 , 2.065,
       2.08 , 2.095, 2.11 , 2.125, 2.14 , 2.155, 2.17 , 2.185, 2.2  ,
       2.215, 2.23 , 2.245, 2.26 , 2.275, 2.29 , 2.305, 2.32 , 2.335,
       2.35 , 2.365, 2.38 , 2.395, 2.41 , 2.425, 2.44 , 2.455, 2.47 ,
       2.485, 2.5  ]), 'MEMORY_VOLUME': 1, 'batch_size': 32, 'gamma': 0.95, 'lr': 0.0001, 'eps': 0.4, 'mode': 'zhou', 'target_update_freq': 100, 'memory_size': 1000}, 'own': False}
"""
# print("\n", Environment)