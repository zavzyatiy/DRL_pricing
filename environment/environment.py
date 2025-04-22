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
import inspect, os
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
PL = Environment["plat_model"]
platform = PL(**Environment["plat_params"])
DIFF_PL = (str(platform) == "dynamic_weights")
plat_epochs = 1
KNOWLEDGE_HORIZON = 0.05 * T
if DIFF_PL:
    Platform_history = []
    Platform_actions = []
    plat_epochs = Environment["plat_params"]["N_epochs"]

def calc_profit_no_plat(gamma, p, theta_d, doli, c_i,
                        inv, x_t, h_plus, v_minus, C, boosting):
    pi = ((1 - gamma) * p - theta_d) * doli
    pi -= c_i * (inv - x_t)
    pi += -h_plus * np.maximum(0, inv - doli) + v_minus * np.minimum(0, inv - doli)
    return (pi, 0)

def calc_profit_with_plat(gamma, p, theta_d, doli, c_i,
                          inv, x_t, h_plus, v_minus, C, boosting):
    demand = doli * boosting
    pi = ((1 - gamma) * p - theta_d) * demand
    pi -= c_i * (inv - x_t)
    pi += -h_plus * np.maximum(0, inv - demand) + v_minus * np.minimum(0, inv - demand)
    pi_plat = (gamma * p + 0.8 * theta_d) * demand
    pi_plat += 0.5 * (h_plus * np.maximum(0, inv - demand) - v_minus * np.minimum(0, inv - demand)) / max(h_plus, v_minus)
    pi_plat = np.sum(pi_plat)
    return (pi, pi_plat)


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
    # if PLATFORM:
    PL = Environment["plat_model"]
    platform = PL(**Environment["plat_params"])
    
    if PLATFORM:
        profit_func = calc_profit_with_plat
    else:
        profit_func = calc_profit_no_plat

    ### Инициализация основного цикла
    if str(firms[0]) == "TN_DDQN":
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

            plat_info = {
                "doli": doli,
                "p": p,
                "stock": np.array(x_t),
                "inv": inv,
            }
            first, second, w, boosting = platform.suggest(plat_info)
            pi, pi_plat = profit_func(gamma, p, theta_d, doli, c_i, inv, np.array(x_t), h_plus, v_minus, C, boosting)
            plat_info = {
                "doli": doli,
                "p": p,
                "boosting": boosting,
                "first": first,
                "second": second,
                "stock": np.array(x_t),
                "inv": inv,
                "action": w,
                "plat_pi": pi_plat,
            }
            demand = doli * boosting
            platform.cache_data(plat_info)

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
                        'current_inventory': max(0, inv[i] - demand[i]),
                        'competitors_prices': new,
                    }

                    firms[i].cache_experience(prev_state, idxs[i], pi[i], new_state)
                    
            x_t = np.maximum(0, inv - demand)

            for i in range(n):
                firms[i].update()
            
            if 0 < t and t % plat_epochs == 0:
                platform.update()

            mem = learn.copy()

            if env == ENV - 1 or (env != ENV - 1 and t >= T - int(1 * KNOWLEDGE_HORIZON)):
                raw_profit_history.append(pi)
                raw_price_history.append(p)
                raw_stock_history.append(inv)
                if DIFF_PL:
                    raw_platform_history.append((pi_plat, w))

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

                    plat_info = {
                        "doli": doli,
                        "p": p,
                        "stock": np.array(x_t),
                        "inv": inv,
                    }
                    first, second, w, boosting = platform.suggest(plat_info)
                    pi, pi_plat = profit_func(gamma, p, theta_d, doli, c_i, inv, np.array(x_t), h_plus, v_minus, C, boosting)
                    plat_info = {
                        "doli": doli,
                        "p": p,
                        "boosting": boosting,
                        "first": first,
                        "second": second,
                        "stock": np.array(x_t),
                        "inv": inv,
                        "action": w,
                        "plat_pi": pi_plat,
                    }
                    demand = doli * boosting
                    platform.cache_data(plat_info)
                    
                    if len(learn) == MEMORY_VOLUME:
                        for i in range(n):
                            new = learn.copy()
                            if len(new) == MEMORY_VOLUME and not(own):
                                for j in range(MEMORY_VOLUME):
                                    new[j] = new[j][: i] + new[j][i + 1 :]

                            new_state = {
                                'current_inventory': max(0, inv[i] - demand[i]),
                                'competitors_prices': new,
                            }

                            firms[i].cache_experience(new_state, idxs[i], pi[i])
                            
                    x_t = np.maximum(0, inv - demand)
                    
                    mem = learn.copy()

                    pbar.update(1)

                    if env == ENV - 1 or (env != ENV - 1 and t >= T - int(1 * KNOWLEDGE_HORIZON)):
                        raw_profit_history.append(pi)
                        raw_price_history.append(p)
                        raw_stock_history.append(inv)
                        if DIFF_PL:
                            raw_platform_history.append((pi_plat, w))
                
                if total_t < 0:
                    total_t = 0
                elif min(total_t + N_epochs, T) < T:
                    for i in range(n):
                        firms[i].update()
                    platform.update()
                    total_t = min(total_t + N_epochs, T)
                else:
                    total_t = min(total_t + N_epochs, T)
                
    elif str(firms[0]) == "PPO_C":
        total_t = -MEMORY_VOLUME
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

                    plat_info = {
                        "doli": doli,
                        "p": p,
                        "stock": np.array(x_t),
                        "inv": inv,
                    }
                    first, second, w, boosting = platform.suggest(plat_info)
                    pi, pi_plat = profit_func(gamma, p, theta_d, doli, c_i, inv, np.array(x_t), h_plus, v_minus, C, boosting)
                    plat_info = {
                        "doli": doli,
                        "p": p,
                        "boosting": boosting,
                        "first": first,
                        "second": second,
                        "stock": np.array(x_t),
                        "inv": inv,
                        "action": w,
                        "plat_pi": pi_plat,
                    }
                    demand = doli * boosting
                    platform.cache_data(plat_info)
                    
                    if len(learn) == MEMORY_VOLUME:
                        for i in range(n):
                            new = learn.copy()
                            if len(new) == MEMORY_VOLUME and not(own):
                                for j in range(MEMORY_VOLUME):
                                    new[j] = new[j][: i] + new[j][i + 1 :]

                            new_state = {
                                'current_inventory': max(0, inv[i] - demand[i]),
                                'competitors_prices': new,
                            }

                            firms[i].cache_experience(new_state, iter_probs[i], pi[i])

                    x_t = np.maximum(0, inv - demand)
                    
                    mem = learn.copy()

                    pbar.update(1)

                    if env == ENV - 1 or (env != ENV - 1 and t >= T - int(1 * KNOWLEDGE_HORIZON)):
                        raw_profit_history.append(pi)
                        raw_price_history.append(p)
                        raw_stock_history.append(inv)
                        if DIFF_PL:
                            raw_platform_history.append((pi_plat, w))
                
                if total_t < 0:
                    total_t = 0
                elif min(total_t + N_epochs, T) < T:
                    for i in range(n):
                        firms[i].update()

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

                    plat_info = {
                        "doli": doli,
                        "p": p,
                        "stock": np.array(x_t),
                        "inv": inv,
                    }
                    first, second, w, boosting = platform.suggest(plat_info)
                    pi, pi_plat = profit_func(gamma, p, theta_d, doli, c_i, inv, np.array(x_t), h_plus, v_minus, C, boosting)
                    plat_info = {
                        "doli": doli,
                        "p": p,
                        "boosting": boosting,
                        "first": first,
                        "second": second,
                        "stock": np.array(x_t),
                        "inv": inv,
                        "action": w,
                        "plat_pi": pi_plat,
                    }
                    demand = doli * boosting
                    platform.cache_data(plat_info)

                    if len(learn) == MEMORY_VOLUME:
                        for i in range(n):
                            new = learn.copy()
                            if len(new) == MEMORY_VOLUME and not(own):
                                for j in range(MEMORY_VOLUME):
                                    new[j] = new[j][: i] + new[j][i + 1 :]

                            new_state = {
                                'current_inventory': max(0, inv[i] - demand[i]),
                                'competitors_prices': new,
                            }

                            firms[i].cache_experience(new_state, iter_probs[i], pi[i])
                    
                    x_t = np.maximum(0, inv - demand)
                    
                    mem = learn.copy()

                    pbar.update(1)

                    if env == ENV - 1 or (env != ENV - 1 and t >= T - int(1 * KNOWLEDGE_HORIZON)):
                        raw_profit_history.append(pi)
                        raw_price_history.append(p)
                        raw_stock_history.append(inv)
                        if DIFF_PL:
                            raw_platform_history.append((pi_plat, w))
                    
                    if 0 < t and t % plat_epochs == 0:
                        platform.update()
                
                if total_t < 0:
                    total_t = 0
                elif min(total_t + N_epochs, T) < T:
                    for i in range(n):
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
    
    Price_history.append(tuple([np.mean(raw_price_history[- int(KNOWLEDGE_HORIZON):, i]) for i in range(n)]))
    Profit_history.append(tuple([np.mean(raw_profit_history[- int(KNOWLEDGE_HORIZON):, i]) for i in range(n)]))
    if SHOW_PROM_RES:
        print("\n", Price_history[-1])
    if HAS_INV == 1:
        Stock_history.append(tuple([np.mean(raw_stock_history[- int(KNOWLEDGE_HORIZON):, i]) for i in range(n)]))
        print(Stock_history[-1])
    if DIFF_PL:
        Platform_history.append(np.mean(raw_platform_history[- int(KNOWLEDGE_HORIZON):, 0]))
        Platform_actions.append(np.mean(raw_platform_history[- int(KNOWLEDGE_HORIZON):, 1]))
        print((Platform_history[-1], Platform_actions[-1]))
    if SHOW_PROM_RES:
        print(Profit_history[-1])
        print("-"*100)
        print("\n")


if VISUALIZE or SAVE:

    if T == 10**6:
        sgladit = int(10**4)
    else:
        sgladit = int(KNOWLEDGE_HORIZON)
    
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
            all_d = C * all_d[:, 1:]
            c = c_i * np.ones(all_mv.shape)
            smoothed_pi = (all_mv - c) * all_d
        else:
            if not DIFF_PL:
                zeros_column = a * np.ones((all_mv.shape[0], 1), dtype=all_mv.dtype)
                all_d = np.hstack((zeros_column, all_mv))
                all_d = np.exp((a-all_d)/mu)
                s = np.sum(all_d, axis = 1)
                all_d = all_d / s[:, np.newaxis]
                all_d = C * all_d[:, 1:]
                stocks = np.concatenate((np.array([[0, 0]]), np.maximum(0, all_inv - all_d)))[:-1]
                smoothed_pi = ((1 - gamma) * all_mv - theta_d) * all_d
                smoothed_pi -= c_i * (all_inv - stocks)
                smoothed_pi += - h_plus * np.maximum(0, all_inv - all_d) + v_minus * np.minimum(0, all_inv - all_d)
            else:
                mv = np.convolve(raw_platform_history[:, 0], kernel, mode='valid')
                smoothed_pi = deepcopy(mv)
                mv = np.convolve(raw_platform_history[:, 1], kernel, mode='valid')
                smoothed_act = deepcopy(mv)
            
        if not DIFF_PL:
            for i in range(n):
                mv = smoothed_pi[:, i]
                if profit_dynamic == "compare":
                    plotThird.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
                else:
                    plotSecond.plot(mv, c = color[i], label = f"Фирма {i + 1}") # , linewidth= 0.2)
        else:
            if profit_dynamic == "compare":
                plotThird.plot(smoothed_pi, label = "Прибыль", color = "#FF7F00") # , linewidth= 0.2)
                ax2 = plotThird.twinx()
                ax2.plot(smoothed_act, label = "Коэфф., %", color = "#1874CD")
            else:
                plotSecond.plot(smoothed_pi, label = "Прибыль", color = "#FF7F00") # , linewidth= 0.2)
                ax2 = plotSecond.twinx()
                ax2.plot(smoothed_act, label = "Коэфф., %", color = "#1874CD")

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
                plotSecond.set_title("Динамика прибыли платформы и коэфф. бустинга")
                plotSecond.set_ylabel(f'Сглаженная прибыль (скользящее среднее по {window_size})')
                ax2.set_ylabel(f'Сглаженный коэфф. (скользящее среднее по {window_size})')
                plotSecond.set_xlabel('Итерация')
            else:
                plotThird.set_title("Динамика прибыли платформы и коэфф. бустинга")
                plotThird.set_ylabel(f'Сглаженная прибыль (скользящее среднее по {window_size})')
                ax2.set_ylabel(f'Сглаженный коэфф. (скользящее среднее по {window_size})')
                plotThird.set_xlabel('Итерация')


if SUMMARY:
    print("\n")
    Price_history = np.array(Price_history)
    Profit_history = np.array(Profit_history)

    print(f"Средняя цена по последним {int(KNOWLEDGE_HORIZON)} раундов:", " ".join([str(round(np.mean(Price_history[:, i]), 3)) for i in range(n)]))

    if HAS_INV == 1:
        Stock_history = np.array(Stock_history)
        print(f"Среднии запасы по последним {int(KNOWLEDGE_HORIZON)} раундов:", " ".join([str(round(np.mean(Stock_history[:, i]), 3)) for i in range(n)]))

    if DIFF_PL:
        Platform_history = np.array(Platform_history)
        print(f"Средняя прибыль платформы по последним {int(KNOWLEDGE_HORIZON)} раундов:", str(round(np.mean(Platform_history), 3)))
        Platform_history = np.array(Platform_history)
        print(f"Средний коэфф. значимости цены для бустинга по последним {int(KNOWLEDGE_HORIZON)} раундов:", str(round(np.mean(Platform_actions), 3)))

    print(f"Средняя прибыль по последним {int(KNOWLEDGE_HORIZON)} раундов:", " ".join([str(round(np.mean(Profit_history[:, i]), 3)) for i in range(n)]))

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
        dest = str(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) + "/simulation_results/"
        for f in os.listdir(dest):
            if not("." in str(f)) and (str(firms[0]) + "_" + num in str(f)):
                folders.append(f)

        res_name = dest + f"/{str(firms[0])}_{num}_{len(folders) + 1}/"

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
        if DIFF_PL:
            path = os.path.join(res_name, "Platform_history.npy")
            np.save(path, Platform_history)
            path = os.path.join(res_name, "Platform_actions.npy")
            np.save(path, Platform_actions)

        with open(res_name + "summary.txt", "w+", encoding="utf-8") as f:
            A = ""
            A = A + f"Средняя цена по последним {int(KNOWLEDGE_HORIZON)} раундов: " + " ".join([str(round(np.mean(Price_history[:, i]), 3)) for i in range(n)])
            A = A + "\n"
            if HAS_INV == 1:
                A = A + f"Среднии запасы по последним {int(KNOWLEDGE_HORIZON)} раундов: " + " ".join([str(round(np.mean(Stock_history[:, i]), 3)) for i in range(n)])
                A = A + "\n"
            
            if DIFF_PL:
                A = A + f"Средняя прибыль платформы по последним {int(KNOWLEDGE_HORIZON)} раундов: " + str(round(np.mean(Platform_history), 3))
                A = A + "\n"
                A = A + f"Средний коэфф. значимости цены для бустинга по последним {int(KNOWLEDGE_HORIZON)} раундов: " + str(round(np.mean(Platform_actions), 3))
                A = A + "\n"

            A = A + f"Средняя прибыль по последним {int(KNOWLEDGE_HORIZON)} раундов: " + " ".join([str(round(np.mean(Profit_history[:, i]), 3)) for i in range(n)])
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