### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt
### для сервака: source /mnt/data/venv_new/bin/activate
### для сервака: python3 environment/environment.py

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
# Вознаграждение платформы
gamma = Environment["gamma"]
# Издержки на доставку товара
theta_d = Environment["theta_d"]
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
elif str(M(**firm_params)) == "SAC":
    MEMORY_VOLUME = Environment["MEMORY_VOLUME"]
    own = Environment["own"]
    batch_size = Environment["firm_params"]["batch_size"]
    N_epochs = Environment["firm_params"]["N_epochs"]
    epochs = Environment["firm_params"]["epochs"]
    assert (N_epochs >= batch_size)

# Инициализация памяти о сложившихся равновесиях
Price_history = []
Profit_history = []
if HAS_INV == 1:
    Stock_history = []

# Рассчет равновесий в случае конкуренции (равновесие по Нэшу)
# и в случае полного картеля
demand_params = Environment["demand_params"]
spros = demand_function(**demand_params)
if VISUALIZE_THEORY:
    p_NE, p_M, pi_NE, pi_M = spros.get_theory(c_i, gamma, theta_d)
    inv_NE, inv_M = spros.distribution([p_NE]*n)[0], spros.distribution([p_M]*n)[0]
C = demand_params["C"]

# Есть ли на рынке платформа?
PLATFORM = Environment["PLATFORM"]
if PLATFORM:
    PL = Environment["plat_model"]
    platform = PL(**Environment["plat_params"])
    DIFF_PL = (str(platform) == "dynamic_weights")
    if DIFF_PL:
        Platform_history = []
else:
    platform = "None"
    DIFF_PL = False

for env in range(ENV):

    # Промежуточные данные о сложившихся равновесиях
    raw_price_history = []
    raw_profit_history = []
    if HAS_INV == 1:
        raw_stock_history = []
    if DIFF_PL:
        raw_platform_history = []

    ### Инициализация однородных фирм
    firms = [deepcopy(M(**firm_params)) for i in range(n)]
    # Память с прошлого хода
    mem = []
    # Инициализация запасов на складах
    if HAS_INV == 1:
        x_t = np.array([0 for i in range(n)])
    
    ### Инициализация платформы
    if PLATFORM:
        PL = Environment["plat_model"]
        platform = PL(**Environment["plat_params"])

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

            if not PLATFORM:
                pi = ((1 - gamma) * p - theta_d) * doli
                pi -= c_i * (inv - np.array(x_t))
                pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
            else:
                boosting = platform.suggest(p)
                pi = ((1 - gamma) * p - theta_d) * doli * boosting
                pi -= c_i * (inv - np.array(x_t))
                pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                platform.cache_data(doli)

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

                        if t >= 0:
                            idxs_i = firms[i].suggest_actions(firm_state)
                        else:
                            idxs_i = (random.sample([i for i in range(len(inventory))], 1)[0],
                                      random.sample([i for i in range(len(prices))], 1)[0])
                            
                        idxs.append(idxs_i)

                    learn = mem.copy()

                    if len(learn) < MEMORY_VOLUME:
                        learn.append([x[1] for x in idxs])
                    else:
                        learn = learn[1:] + [[x[1] for x in idxs]]
                    
                    inv = np.array([inventory[x[0]] for x in idxs])
                    p = np.array([prices[x[1]] for x in idxs])

                    doli = spros.distribution(p)

                    if not PLATFORM:
                        pi = ((1 - gamma) * p - theta_d) * doli
                        pi -= c_i * (inv - np.array(x_t))
                        pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                    else:
                        boosting = platform.suggest(p)
                        pi = ((1 - gamma) * p - theta_d) * doli * boosting
                        pi -= c_i * (inv - np.array(x_t))
                        pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                        platform.cache_data(doli)
                    
                    # ### БЫЛО
                    # if len(learn) == MEMORY_VOLUME:
                    #     for i in range(n):
                    #         state_i = mem.copy()
                    #         if len(state_i) == MEMORY_VOLUME and not(own):
                    #             for j in range(MEMORY_VOLUME):
                    #                 state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                            
                    #         new = learn.copy()
                    #         if len(new) == MEMORY_VOLUME and not(own):
                    #             for j in range(MEMORY_VOLUME):
                    #                 new[j] = new[j][: i] + new[j][i + 1 :]
                            
                    #         prev_state = {
                    #             'current_inventory': x_t[i],
                    #             'competitors_prices': state_i,
                    #         }

                    #         new_state = {
                    #             'current_inventory': max(0, inv[i] - doli[i]),
                    #             'competitors_prices': new,
                    #         }

                    #         firms[i].cache_experience(prev_state, idxs[i], pi[i], new_state)

                    if len(learn) == MEMORY_VOLUME:
                        for i in range(n):
                            new = learn.copy()
                            if len(new) == MEMORY_VOLUME and not(own):
                                for j in range(MEMORY_VOLUME):
                                    new[j] = new[j][: i] + new[j][i + 1 :]

                            new_state = {
                                'current_inventory': max(0, inv[i] - doli[i]),
                                'competitors_prices': new,
                            }

                            firms[i].cache_experience(new_state, idxs[i], pi[i])
                            
                    x_t = np.maximum(0, inv - doli)
                    
                    mem = learn.copy()

                    pbar.update(1)
                    raw_profit_history.append(pi)
                    raw_price_history.append(p)
                    raw_stock_history.append(inv)
                
                if total_t < 0:
                    total_t = 0
                elif min(total_t + N_epochs, T) < T:
                    for i in range(n):
                        firms[i].update()   
                    total_t = min(total_t + N_epochs, T)
                else:
                    total_t = min(total_t + N_epochs, T)
                
    elif str(firms[0]) == "PPO_C":
        total_t = -MEMORY_VOLUME
        count_plat = 0
        with tqdm(total = T + MEMORY_VOLUME, desc=f'Раунд {env + 1}') as pbar:
            while total_t < T:
                if total_t < 0:
                    min_t = total_t
                    max_t = 0
                else:
                    min_t = total_t
                    max_t = min(total_t + N_epochs, T)
                
                for t in range(min_t, max_t):

                    acts = []
                    iter_probs = []
                    for i in range(n):
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
                        
                        iter_probs.append((u_inv, u_prc))
                        acts.append(acts_i)

                    learn = mem.copy()

                    if len(learn) < MEMORY_VOLUME:
                        learn.append([x[1] for x in acts])
                    else:
                        learn = learn[1:] + [[x[1] for x in acts]]
                    
                    inv = np.array([x[0] for x in acts])
                    p = np.array([x[1] for x in acts])

                    doli = spros.distribution(p)

                    if not PLATFORM:
                        pi = ((1 - gamma) * p - theta_d) * doli
                        pi -= c_i * (inv - np.array(x_t))
                        pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                    else:
                        # boosting = platform.suggest(p)
                        # demand = doli * boosting.detach().numpy()
                        # pi = ((1 - gamma) * p - theta_d) * demand
                        # pi -= c_i * (inv - np.array(x_t))
                        # pi += -h_plus * np.maximum(0, inv - demand) + v_minus * np.minimum(0, inv - demand)
                        first, second, boosting = platform.suggest(p)
                        demand = doli * boosting
                        pi = ((1 - gamma) * p - theta_d) * demand
                        pi -= c_i * (inv - np.array(x_t))
                        pi += -h_plus * np.maximum(0, inv - demand) + v_minus * np.minimum(0, inv - demand)
                        # if t == max_t - 1:
                        pi_plat = (gamma * p + theta_d) * demand
                        pi_plat += (h_plus * np.maximum(0, inv - demand) - v_minus * np.minimum(0, inv - demand))/C
                        pi_plat = np.sum(pi_plat)
                        # print(pi_plat)
                        plat_info = {
                            "boosting": boosting,
                            "doli": doli,
                            "demand": demand,
                            "price_val": first,
                            "inv_val": second,
                            'current_inventory': inv,
                            "competitors_prices": p,
                            "plat_pi": pi_plat,
                            "timestamp": max_t - 1 - t + t * int(count_plat %2 != 1),
                        }
                        platform.cache_data(plat_info)
                    
                    ### БЫЛО:
                    # if len(learn) == MEMORY_VOLUME:
                    #     for i in range(n):
                    #         state_i = mem.copy()
                    #         if len(state_i) == MEMORY_VOLUME and not(own):
                    #             for j in range(MEMORY_VOLUME):
                    #                 state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                            
                    #         new = learn.copy()
                    #         if len(new) == MEMORY_VOLUME and not(own):
                    #             for j in range(MEMORY_VOLUME):
                    #                 new[j] = new[j][: i] + new[j][i + 1 :]
                            
                    #         prev_state = {
                    #             'current_inventory': x_t[i],
                    #             'competitors_prices': state_i,
                    #         }

                    #         new_state = {
                    #             'current_inventory': max(0, inv[i] - doli[i]),
                    #             'competitors_prices': new,
                    #         }

                    #         firms[i].cache_experience(prev_state, iter_probs[i], pi[i], new_state)
                        
                    ### СТАЛО:
                    if len(learn) == MEMORY_VOLUME:
                        for i in range(n):
                            new = learn.copy()
                            if len(new) == MEMORY_VOLUME and not(own):
                                for j in range(MEMORY_VOLUME):
                                    new[j] = new[j][: i] + new[j][i + 1 :]

                            new_state = {
                                'current_inventory': max(0, inv[i] - doli[i]),
                                'competitors_prices': new,
                            }

                            firms[i].cache_experience(new_state, iter_probs[i], pi[i])

                    x_t = np.maximum(0, inv - doli)
                    
                    mem = learn.copy()

                    pbar.update(1)
                    raw_profit_history.append(pi)
                    raw_price_history.append(p)
                    raw_stock_history.append(inv)
                    if DIFF_PL:
                        raw_platform_history.append(pi_plat)
                
                if total_t < 0:
                    total_t = 0
                elif min(total_t + N_epochs, T) < T:
                    for i in range(n):
                        firms[i].update()
                    count_plat += 1
                    if PLATFORM and count_plat %2 == 0:
                        platform.update()
                    total_t = min(total_t + N_epochs, T)
                else:
                    total_t = min(total_t + N_epochs, T)

    elif str(firms[0]) == "SAC":
        total_t = -MEMORY_VOLUME
        # total_t = -MEMORY_VOLUME - batch_size
        with tqdm(total = T + MEMORY_VOLUME, desc=f'Раунд {env + 1}') as pbar:
        # with tqdm(total = T + MEMORY_VOLUME + batch_size, desc=f'Раунд {env + 1}') as pbar:
            while total_t < T:
                if total_t < 0:
                    min_t = total_t
                    max_t = 0
                else:
                    min_t = total_t
                    max_t = min(total_t + N_epochs, T)
        
                for t in range(min_t, max_t):
                # for t in tqdm(range(- MEMORY_VOLUME - batch_size, T), f"Раунд {env + 1}"):
                    acts = []
                    iter_probs = []
                    
                    for i in range(n):
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
                            u_inv = torch.distributions.Normal(0, 1).rsample()
                            u_prc = torch.distributions.Normal(0, 1).rsample()
                            act_inv = x_t[i] + torch.sigmoid(u_inv/10) * (inventory[1] - x_t[i])
                            act_price = prices[0] + torch.sigmoid(u_prc/10) * (prices[1] - prices[0])
                            acts_i = (act_inv, act_price)
                        
                        iter_probs.append((u_inv, u_prc))
                        acts.append(acts_i)

                    learn = mem.copy()

                    if len(learn) < MEMORY_VOLUME:
                        learn.append([x[1] for x in acts])
                    else:
                        learn = learn[1:] + [[x[1] for x in acts]]
                    
                    inv = np.array([x[0] for x in acts])
                    p = np.array([x[1] for x in acts])

                    doli = spros.distribution(p)

                    if not PLATFORM:
                        pi = ((1 - gamma) * p - theta_d) * doli
                        pi -= c_i * (inv - np.array(x_t))
                        pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                    else:
                        boosting = platform.suggest(p)
                        pi = ((1 - gamma) * p - theta_d) * doli * boosting
                        pi -= c_i * (inv - np.array(x_t))
                        pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
                        platform.cache_data(doli)

                    # if len(learn) == MEMORY_VOLUME:
                    #     for i in range(n):
                    #         state_i = mem.copy()
                    #         if len(state_i) == MEMORY_VOLUME and not(own):
                    #             for j in range(MEMORY_VOLUME):
                    #                 state_i[j] = state_i[j][: i] + state_i[j][i + 1 :]
                            
                    #         new = learn.copy()
                    #         if len(new) == MEMORY_VOLUME and not(own):
                    #             for j in range(MEMORY_VOLUME):
                    #                 new[j] = new[j][: i] + new[j][i + 1 :]
                            
                    #         prev_state = {
                    #             'current_inventory': x_t[i],
                    #             'competitors_prices': state_i,
                    #         }

                    #         new_state = {
                    #             'current_inventory': max(0, inv[i] - doli[i]),
                    #             'competitors_prices': new,
                    #         }
                            
                    #         firms[i].cache_experience(prev_state, iter_probs[i], pi[i], new_state)

                    if len(learn) == MEMORY_VOLUME:
                        for i in range(n):
                            new = learn.copy()
                            if len(new) == MEMORY_VOLUME and not(own):
                                for j in range(MEMORY_VOLUME):
                                    new[j] = new[j][: i] + new[j][i + 1 :]

                            new_state = {
                                'current_inventory': max(0, inv[i] - doli[i]),
                                'competitors_prices': new,
                            }

                            firms[i].cache_experience(new_state, iter_probs[i], pi[i])
                    
                    x_t = np.maximum(0, inv - doli)
                    
                    mem = learn.copy()

                    pbar.update(1)
                    raw_profit_history.append(pi)
                    raw_price_history.append(p)
                    raw_stock_history.append(inv)
                
                if total_t < 0:
                    total_t = 0
                elif min(total_t + N_epochs, T) < T:
                    # print("#"*50)
                    for i in range(n):
                        # print("Обновление фирмы", i)
                        firms[i].update()   
                    total_t = min(total_t + N_epochs, T)
                else:
                    total_t = min(total_t + N_epochs, T)

    raw_price_history = np.array(raw_price_history)
    raw_profit_history = np.array(raw_profit_history)
    if HAS_INV == 1:
        raw_stock_history = np.array(raw_stock_history)
    if DIFF_PL:
        raw_platform_history = np.array(raw_platform_history)

    Price_history.append(tuple([np.mean(raw_price_history[-int(T/20):, i]) for i in range(n)]))
    Profit_history.append(tuple([np.mean(raw_profit_history[-int(T/20):, i]) for i in range(n)]))
    if SHOW_PROM_RES:
        print("\n", Price_history[-1])
    if HAS_INV == 1:
        Stock_history.append(tuple([np.mean(raw_stock_history[-int(T/20):, i]) for i in range(n)]))
        print(Stock_history[-1])
    if DIFF_PL:
        Platform_history.append(np.mean(raw_platform_history[-int(T/20):]))
        print(Platform_history[-1])
    if SHOW_PROM_RES:
        print(Profit_history[-1])
        print("-"*100)
        print("\n")


if VISUALIZE or SAVE:

    if T == 10**6:
        sgladit = int(10**4)
    else:
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
            if not DIFF_PL:
                zeros_column = a * np.ones((all_mv.shape[0], 1), dtype=all_mv.dtype)
                all_d = np.hstack((zeros_column, all_mv))
                all_d = np.exp((a-all_d)/mu)
                s = np.sum(all_d, axis = 1)
                all_d = all_d / s[:, np.newaxis]
                all_d = all_d[:, 1:]
                stocks = np.concatenate((np.array([[0, 0]]), np.maximum(0, all_inv - all_d)))[:-1]
                smoothed_pi = ((1 - gamma) * all_mv - theta_d) * all_d
                smoothed_pi -= c_i * (all_inv - stocks)
                smoothed_pi += - h_plus * np.maximum(0, all_inv - all_d) + v_minus * np.minimum(0, all_inv - all_d)
            else:
                mv = np.convolve(raw_platform_history, kernel, mode='valid')
                smoothed_pi = deepcopy(mv)
            
        if not DIFF_PL:
            for i in range(n):
                mv = smoothed_pi[:, i]
                if profit_dynamic == "compare":
                    plotThird.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
                else:
                    plotSecond.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
        else:
            if profit_dynamic == "compare":
                plotThird.plot(smoothed_pi) #, label = "Платформа") # , linewidth= 0.2)
            else:
                plotSecond.plot(smoothed_pi) #, label = "Платформа") # , linewidth= 0.2)

        if not DIFF_PL:
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
        else:
            if profit_dynamic != "compare":
                plotSecond.set_title("Динамика прибыли платформы")
                plotSecond.set_ylabel(f'Сглаженная прибыль (скользящее среднее по {window_size})')
                plotSecond.set_xlabel('Итерация')
            else:
                plotThird.set_title("Динамика прибыли платформы")
                plotThird.set_ylabel(f'Сглаженная прибыль (скользящее среднее по {window_size})')
                plotThird.set_xlabel('Итерация')


if SUMMARY:
    print("\n")
    Price_history = np.array(Price_history)
    Profit_history = np.array(Profit_history)

    print(f"Средняя цена по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Price_history[:, i]), 3)) for i in range(n)]))

    if HAS_INV == 1:
        Stock_history = np.array(Stock_history)
        print(f"Среднии запасы по последним {int(T/20)} раундов:", " ".join([str(round(np.mean(Stock_history[:, i]), 3)) for i in range(n)]))

    if DIFF_PL:
        Platform_history = np.array(Platform_history)
        print(f"Средняя прибыль платформы по последним {int(T/20)} раундов:", str(round(np.mean(Platform_history), 3)))

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
        num = Environment["folder_num"]
        for f in os.listdir("./DRL_pricing/environment/simulation_results/"):
            if not("." in str(f)) and (str(firms[0]) + "_" + num in str(f)):
                folders.append(f)

        res_name = f"./DRL_pricing/environment/simulation_results/{str(firms[0])}_{num}_{len(folders) + 1}/"
        
        if not os.path.exists(res_name):
            os.makedirs(res_name)

        with open(res_name + "params.txt", "w+", encoding="utf-8") as f:
            to_write = deepcopy(Environment)
            to_write["firm_model"] = str(firms[0])
            to_write["plat_model"] = str(platform)
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

            if VISUALIZE_THEORY:
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


# print("\n", Environment)