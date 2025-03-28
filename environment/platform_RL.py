### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requiremnts.txt

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random
from copy import deepcopy

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class fixed_weights:
    def __init__(
            self,
            weights: list,
            memory_size: int,
            n: int,
            p_inf: float,
            p_max: float,
            C: float,
			):
        
        weights = np.array(weights)
        assert np.sum(weights) == 1

        self.weights = weights
        self.memory_size = memory_size
        self.p = p_max
        self.diff = p_max - p_inf
        self.n = n
        self.C = C
        
        self.memory = []


    def __repr__(self):
        return "fixed_weights"


    def cache_data(self, state):

        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append(state)
    

    def suggest(self, p):
        first = (self.p - p)/self.diff
        if len(self.memory) > 0:
            second = np.mean(np.array(self.memory).T, axis = 1) / self.C
        else:
            second = np.ones(self.n)
        res = self.weights[0] * first + self.weights[1] * second
        e_res = np.exp(res)
        return self.n * e_res/np.sum(e_res)


    def update(self):
        pass

