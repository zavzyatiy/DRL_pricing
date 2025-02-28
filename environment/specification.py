### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random
import numpy as np
# import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from firms_RL import epsilon_greedy, TQL

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
    "T": 100000,
    "ENV": 100,
    "n": 2,
    "m": 5,
    "delta": 0.95,
    "gamma": 0.5,
    "c_i": 1,
    "h_plus": 0,
    "v_minus": 0,
    "eta": 0.05,
    "color": ["#FF7F00", "#1874CD", "#548B54", "#CD2626", "#CDCD00"],
    "profit_dynamic": "real", # "MA", "real", "compare"
    "VISUALIZE_THEORY": True,
    "VISUALIZE": False,
    "SAVE": True,
}
e2 = {
    "p_inf": e1["c_i"],
    "p_sup": 2.5,
    "arms_amo": 51,
}

mode = "D"

if mode == "D":
    prices = np.linspace(e2["p_inf"], e2["p_sup"], e2["arms_amo"])
else:
    prices = (e2["p_inf"], e2["p_sup"])

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

e4 = {
    "prices": prices,
    "firm_model": TQL, # epsilon_greedy
    "firm_params": {
        "eps": 0.5,
        "Q_mat": np.zeros((len(prices)**(MEMORY_VOLUME * (e1["n"] - (1 - int(own)))), len(prices))),
        "MEMORY_VOLUME": MEMORY_VOLUME,
        "n": e1["n"],
        "own": own,
        "index_list": [x for x in range(len(prices)**MEMORY_VOLUME)],
        "action_list": prices,
        "delta": e1["delta"],
        "alpha": 0.15,
        "mode": "zhou", # None, "sanchez_cartas", "zhou"
    },
}

e5 = {

}

Environment = e1 | e2 | e3 | e4 | e5

# a = [1, 2]
# print(a[:0] + a[1:], a[:1] + a[2:])