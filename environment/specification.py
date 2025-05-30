### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt
### для сервака: source /mnt/data/venv_new/bin/activate
### для сервака: python3 environment/environment.py

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from firms_RL import epsilon_greedy, TQL, TN_DDQN, PPO_D, PPO_C, SAC
from platform_RL import no_platform, fixed_weights, dynamic_weights

### Модель спроса
class demand_function:
     
    def __init__(
            self,
            n: int,
            mode: str,
            a: None,
            mu: None,
            C: None,
			):

        mode_list = ["logit"]
        assert n >= 2, "Demand is built for olygopoly, so n > 2!"
        assert mode in mode_list, f"Demand function must be in [{' '.join(mode_list)}]"
        self.n = n
        self.mode = mode
        if a and mu and C:
            self.a = a
            self.mu = mu
            self.C = C
    
    def distribution(self, prices):
        assert len(prices) == self.n, f"Demand is built for n = {self.n}, not for {len(prices)}"

        if self.mode == "logit":
            # s = list(prices)
            if self.a:
                s = np.concatenate(([self.a], prices))
                # exp_s = [1] + [np.exp((self.a - x)/self.mu) for x in s]
                exp_s = np.exp((self.a - s)/self.mu)
            else:
                s = np.concatenate(([0], prices))
                # exp_s = [1] + [np.exp(-x) for x in s]
                exp_s = np.exp((self.a - s)/self.mu)
            
            sum_exp = np.sum(exp_s)
            # return [x/sum_exp for x in exp_s[1:]]
            res = exp_s/sum_exp
            return self.C * res[1:]
    
    def get_theory(self, c_i, gamma, theta_d):
        precision = 10**(-5)
        if self.mode == "logit":
            point_NE = 0
            c = 0
            while c == 0 or abs(point_NE - c) > precision:
                point_NE = c
                c = np.exp((point_NE - self.a)/self.mu)
                c = (c_i + theta_d)/(1 - gamma) + self.mu * (self.n + c)/(self.n + c - 1)
            
            point_M = self.a + self.mu
            c = 0
            while c == 0 or abs(point_M - c) > precision:
                c = point_M
                point_M = -self.mu * np.log(point_M - (c_i + theta_d)/(1 - gamma) - self.mu) + self.mu * np.log(self.mu * self.n) + self.a
            
            pi_NE = ((1 - gamma) * point_NE - theta_d - c_i)*self.distribution([point_NE]*self.n)[0]
            pi_M = ((1 - gamma) * point_M - theta_d - c_i)*self.distribution([point_M]*self.n)[0]

            return point_NE, point_M, pi_NE, pi_M

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

e1 = {
    "T": 100000,         # 100000, 200000
    "ENV": 15,
    "n": 2,
    "m": 30,
    "delta": 0.95,      # 0.95, 0.99
    "gamma": 0.1,
    "theta_d": 0.043,
    "c_i": 1,           # 0.25, 1
    "h_plus": 3,        # Из Zhou: примерно половина монопольной цены
    "v_minus": 3,       # Из Zhou: примерно четверть монопольной цены
    "eta": 0.05,
    "color": ["#FF7F00", "#1874CD", "#548B54", "#CD2626", "#CDCD00"],
    "profit_dynamic": "compare", # "MA", "real", "compare"
    "loc": "lower left",
    "VISUALIZE_THEORY": True,
    "VISUALIZE": True,
    "SAVE": True,
    "SUMMARY": True,
    "SHOW_PROM_RES": True,
    "SAVE_SUMMARY": True,
    "RANDOM_SEED": 34,
}

# Это чтобы я случайно не потерял все результаты симуляций
e1["SAVE_SUMMARY"] = e1["SAVE_SUMMARY"] or ((e1["ENV"] >= 10) and (e1["T"] >= 10000))

e2 = {
    "p_inf": (e1["c_i"] + e1["theta_d"]) / (1 - e1["gamma"]),
    "p_sup": (e1["c_i"] + e1["theta_d"]) / (1 - e1["gamma"]) + 1.5,    # 2, 2.5
    "arms_amo_price": 101,   # 21, 101
    "arms_amo_inv": 101,     # 21, 101
}

e3 = {
    "demand_params":{
        "n": e1["n"],
        "mode": "logit",
        "a": e2["p_inf"] + 1,
        "mu": 0.25,
        "C": 36,
    },
}

mode = "D" # C, D

if mode == "D":
    prices = np.linspace(e2["p_inf"], e2["p_sup"], e2["arms_amo_price"])
    inventory = np.linspace(0, e3["demand_params"]["C"], e2["arms_amo_inv"])
else:
    prices = (e2["p_inf"], e2["p_sup"])
    inventory = (0, e3["demand_params"]["C"])

MEMORY_VOLUME = 1
own = False
ONLY_OWN = False

##########################
### TN_DDQN
##########################
assert mode == "D"
e4 = {
    "prices": prices,
    "inventory": inventory,
    "firm_model": TN_DDQN,
    "firm_params": {
        "state_dim": 1 + MEMORY_VOLUME * (e1["n"] - (1 - int(own))),
        "inventory_actions": inventory,
        "price_actions": prices,
        "MEMORY_VOLUME": MEMORY_VOLUME,
        "batch_size": 32, # 32
        "gamma": e1["delta"],
        "lr": 0.0001,
        "eps": 0.4,
        "mode": "zhou", # None, "sanchez_cartas", "zhou"
        "target_update_freq": 100, # e1["T"]//100, 100
        "memory_size": 1000, # 10000
        "cuda_usage": True,
        "eps_min": 0.01,
        "eps_max": 1,
        "beta": 1.5/(10**4),
    },
    "own": own,
}
##########################
### PPO-D
##########################
# assert mode == "D"
# e4 = {
#     "prices": prices,
#     "inventory": inventory,
#     "firm_model": PPO_D,
#     "firm_params": {
#         "state_dim": 1 + MEMORY_VOLUME * (e1["n"] - (1 - int(own))),
#         "inventory_actions": inventory,
#         "price_actions": prices,
#         "batch_size": 128,          # 32, 64, 100, 128
#         "N_epochs": 256,            # 100, 200, e1["T"]//100
#         "epochs": 10,               # 25
#         "gamma": e1["delta"],
#         "actor_lr": 1.5 * 1e-4,
#         "critic_lr": 1.5 * 1e-4,
#         "clip_eps": 0.2,
#         "lmbda": 1,
#         "cuda_usage": False,
#     },
#     "MEMORY_VOLUME": MEMORY_VOLUME,
#     "own": own,
# }
##########################
### PPO-C
##########################
# assert mode == "C"
# e4 = {
#     "prices": prices,
#     "inventory": inventory,
#     "firm_model": PPO_C,
#     "firm_params": {
#         "state_dim": 1 + MEMORY_VOLUME * (e1["n"] - (1 - int(own))),
#         "inventory_actions": inventory,
#         "price_actions": prices,
#         "batch_size": 128,          # 32, 64, 100, 128
#         "N_epochs": 256,            # 100, 200, e1["T"]//100
#         "epochs": 10,               # 25
#         "gamma": e1["delta"],
#         "actor_lr": 1.5 * 1e-4,
#         "critic_lr": 1.5 * 1e-4,
#         "clip_eps": 0.2,
#         "lmbda": 1,
#         "cuda_usage": False,
#     },
#     "MEMORY_VOLUME": MEMORY_VOLUME,
#     "own": own,
# }
##########################
### SAC
##########################
# assert mode == "C"
# e4 = {
#     "prices": prices,
#     "inventory": inventory,
#     "firm_model": SAC,
#     "firm_params": {
#         "state_dim": 1 + MEMORY_VOLUME * (e1["n"] - (1 - int(own))),
#         "inventory_actions": inventory,
#         "price_actions": prices,
#         "batch_size": 100,         # 32, 64, 100, 128
#         "N_epochs": 100,           # 100, 200, e1["T"]//100
#         "epochs": 1,
#         "MC_samples": 200,
#         "gamma": e1["delta"],
#         "actor_lr": 3e-4,
#         "critic_lr": 3e-4,
#         "alpha_lr": 3e-4,
#         "target_entropy": -2,
#         "target_scaling": 1,
#         "tau": 0.05,
#         "cuda_usage": False,
#     },
#     "MEMORY_VOLUME": MEMORY_VOLUME,
#     "own": own,
# }
##########################
### No platform
##########################
# e5 = {
#     "folder_num": "0",
#     "PLATFORM": False,
#     "plat_model": no_platform,
#     "plat_params": {},
# }
##########################
### Fixed weights platform
##########################
# e5 = {
#     "folder_num": "1",
#     "PLATFORM": True,
#     "plat_model": fixed_weights,
#     "plat_params":{
#         "weight": 1/3,
#         "memory_size": e1["m"],
#         "n": e1["n"],
#         "p_inf": e2["p_inf"],
#         "p_max": e2["p_sup"],
#         "C": e3["demand_params"]["C"],
#     }
# }
##########################
### Dynamic platform
##########################
e5 = {
    "folder_num": "2",
    "PLATFORM": True,
    "plat_model": dynamic_weights,
    "plat_params": {
        "state_dim": 2 * MEMORY_VOLUME * e1["n"] + 2 * MEMORY_VOLUME * (e1["n"] - 1),
        "d_memory_size": e1["m"],
        "alpha_actions": np.linspace(0, 1, max(e2["arms_amo_price"], e2["arms_amo_inv"])),
        "n": e1["n"],
        "p_inf": e2["p_inf"],
        "p_max": e2["p_sup"],
        "C": e3["demand_params"]["C"],
        "batch_size": 128,          # 32, 64, 100, 128
        "N_epochs": 256,            # 100, 200, e1["T"]//100
        "epochs": 10,               # 25
        "gamma": e1["delta"],
        "actor_lr": 1.5 * 1e-4,
        "critic_lr": 1.5 * 1e-4,
        "clip_eps": 0.2,
        "lmbda": 1,
        "cuda_usage": False,
        },
}

Environment = e1 | e2 | e3 | e4 | e5

# print(Environment)

# a = np.load(r"DRL_pricing\environment\simulation_results\PPO_D_2_0\Profit_history.npy")
# for b in [["{:.2f}".format(x[0]), "{: .2f}".format(x[1])] for x in a.tolist()]:
#     print(b)