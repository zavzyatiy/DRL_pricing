### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random
import numpy as np
# import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from firms_RL import epsilon_greedy, TQL, TN_DDQN

### Всевозможные модели спросов
class demand_function:
     
    def __init__(
            self,
            n: int,
            mode: str,
            a: None,
            mu: None,          
			):

        mode_list = ["logit"]
        assert n >= 2, "Demand is built for olygopoly, so n > 2!"
        assert mode in mode_list, f"Demand function must be in [{' '.join(mode_list)}]"
        self.n = n
        self.mode = mode
        if a and mu:
            self.a = a
            self.mu = mu
        else:
            self.a = None
            self.mu = None
    
    def distribution(self, prices):
        assert len(prices) == self.n, f"Demand is built for n = {self.n}, not for {len(prices)}"

        if self.mode == "logit":
            s = list(prices)
            if self.a:
                exp_s = [1] + [np.exp((self.a - x)/self.mu) for x in s]
            else:
                exp_s = [1] + [np.exp(-x) for x in s]
            
            sum_exp = sum(exp_s)
            return [x/sum_exp for x in exp_s[1:]]
    
    def get_theory(self, c_i):
        precision = 0.0001
        if self.mode == "logit":
            point_NE = 0
            c = 0
            while c == 0 or abs(point_NE - c) > precision:
                point_NE = c
                c = np.exp((point_NE - self.a)/self.mu)
                c = c_i + self.mu * (self.n + c)/(self.n + c - 1)
            
            point_M = self.a
            c = 0
            while c == 0 or abs(point_M - c) > precision:
                c = point_M
                point_M = -self.mu * np.log(point_M - c_i - self.mu) + self.mu * np.log(self.mu * self.n) + self.a
            
            pi_NE = (point_NE - c_i)*self.distribution([point_NE]*self.n)[0]
            pi_M = (point_M - c_i)*self.distribution([point_M]*self.n)[0]

            return point_NE, point_M, pi_NE, pi_M

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

e1 = {
    "T": 1000,
    "ENV": 1,
    "n": 2,
    "m": 5,
    "delta": 0.95,
    "gamma": 0.5,
    "c_i": 0.25, # 0.25, 1
    "h_plus": 1.17498, # 1.17498/2, # Из Zhou: примерно половина монопольной цены
    "v_minus": 1.17498, # 1.17498/4, # Из Zhou: примерно четверть монопольной цены
    "eta": 0.05,
    "color": ["#FF7F00", "#1874CD", "#548B54", "#CD2626", "#CDCD00"],
    "profit_dynamic": "compare", # "MA", "real", "compare"
    "loc": "lower left",
    "VISUALIZE_THEORY": True,
    "VISUALIZE": True,
    "SAVE": False,
    "SUMMARY": True,
}

e2 = {
    "p_inf": e1["c_i"],
    "p_sup": 2, # 3*e1["c_i"] + e1["h_plus"] + e1["v_minus"], 2.5
    "arms_amo_price": 21,
    "arms_amo_inv": 21,
}

mode = "D"

if mode == "D":
    prices = np.linspace(e2["p_inf"], e2["p_sup"], e2["arms_amo_price"])
    inventory = np.linspace(0, 1, e2["arms_amo_inv"])
else:
    prices = (e2["p_inf"], e2["p_sup"])
    inventory = (0, 1)


e3 = {
    "demand_params":{
        "n": e1["n"],
        "mode": "logit",
        "a": e1["c_i"] + 1,
        "mu": 0.25,
    },
}

MEMORY_VOLUME = 1
own = False
ONLY_OWN = False

##########################
### TQL
##########################
# e4 = {
#     "prices": prices,
#     "firm_model": TQL, # epsilon_greedy
#     "firm_params": {
#         "eps": 0.4,
#         "Q_mat": np.zeros((len(prices)**(MEMORY_VOLUME * (e1["n"] - (1 - int(own)))), len(prices))),
#         "MEMORY_VOLUME": MEMORY_VOLUME,
#         "n": e1["n"],
#         "own": own,
#         "ONLY_OWN": ONLY_OWN,
#         "index_list": [x for x in range(len(prices)**MEMORY_VOLUME)],
#         "action_list": prices,
#         "delta": e1["delta"],
#         "alpha": 0.15,
#         "mode": "zhou", # None, "sanchez_cartas", "zhou"
#     },
# }
##########################
### TN_DDQN
##########################
e4 = {
    "prices": prices,
    "inventory": inventory,
    "firm_model": TN_DDQN,
    "firm_params": {
        "state_dim": 1 + MEMORY_VOLUME * (e1["n"] - (1 - int(own))),
        "inventory_actions": inventory,
        "price_actions": prices,
        "MEMORY_VOLUME": MEMORY_VOLUME,
        "batch_size": 128, # 32
        "gamma": e1["delta"],
        "lr": 0.0001,
        "eps": 0.4,
        "mode": "zhou", # None, "sanchez_cartas", "zhou"
        "target_update_freq": e1["T"]//100, # e1["T"]//100, 100
        "memory_size": 1000, # 10000
    },
    "own": own,
}

e5 = {

}

Environment = e1 | e2 | e3 | e4 | e5

# print(2*0.25 + 1.17498/2 + 1.17498/4)

### Архив возможных параметризаций алгоритмов для фирм:

# mode = None # None, "sanchez_cartas", "zhou"

# firm1 = epsilon_greedy(
#     eps,
#     np.zeros(len(prices)),
#     prices,
#     mode = mode,
#     )

# firm2 = epsilon_greedy(
#     eps,
#     np.zeros(len(prices)),
#     prices,
#     mode = mode,
#     )

bbb = np.array([1, 2, 3])
print(np.concatenate(([0], bbb)))