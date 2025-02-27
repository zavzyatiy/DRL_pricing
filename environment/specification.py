### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random
import numpy as np
# import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

Environment = {
    
}