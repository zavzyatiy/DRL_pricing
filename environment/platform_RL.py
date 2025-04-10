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
            weight: float,
            memory_size: int,
            n: int,
            p_inf: float,
            p_max: float,
            C: float,
			):
        
        # weights = np.array(weights)
        # assert np.sum(weights) == 1

        self.weight = torch.tensor(weight)
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

        self.memory.append(state["demand"])
    

    def suggest(self, p):
        first = (self.p - p)/self.diff
        if len(self.memory) > 0:
            second = np.mean(np.array(self.memory).T, axis = 1) / self.C
        else:
            second = np.ones(self.n)
        # res = self.weights[0] * first + self.weights[1] * second
        # e_res = np.exp(res)
        # return self.n * e_res/np.sum(e_res)
        w = self.weight
        res = w * torch.tensor(first.tolist()) + (1 - w) * torch.tensor(second.tolist())
        e_res = res.exp()
        return self.n * e_res/(e_res).sum()


    def update(self):
        pass


class dynamic_weights:
    def __init__(
            self,
            demand_memory_size: int,
            n: int,
            p_inf: float,
            p_max: float,
            C: float,
            starting_weight: float,
            delta: float,
            lr: float,
            cuda_usage: bool,
            gamma: float,
            theta_d: float,
            h_plus: float,
            v_minus: float,
			):

        self.weight = torch.tensor(starting_weight, requires_grad= True)
        self.demand_memory_size = demand_memory_size
        self.p = p_max
        self.diff = p_max - p_inf
        self.n = n
        self.C = C

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.delta = delta
        self.lr = lr
        self.gamma = gamma
        self.theta_d = theta_d
        self.h_plus = h_plus
        self.v_minus = v_minus

        self.memory = []
        self.res = 0
        self.profit_grad = torch.tensor(0.0)


    def __repr__(self):
        return "dynamic_weights"


    def cache_data(self, state):

        if len(self.memory) >= self.demand_memory_size:
            self.memory.pop(0)
        
        t = state["timestamp"]
        if t < self.demand_memory_size and t > 0:
            # print(t)
            inv = torch.tensor(state['current_inventory'].tolist())
            # prom_memory = [x.clone() for x in self.memory]
            # print(self.memory)
            boosting = state["boosting"]
            demand = torch.tensor(state["demand"])
            p = state["competitors_prices"]
            res = torch.tensor((self.gamma * p + self.theta_d)) * demand * boosting
            res += (self.h_plus * torch.maximum(torch.tensor(0.0), inv - demand))/self.C
            res -= (self.v_minus * torch.minimum(torch.tensor(0.0), inv - demand))/self.C
            res = self.delta**t * res.sum()
            # res = res.sum()
            self.res += res.item()
            res.backward()
            self.profit_grad += self.weight.grad
            # self.memory = [x.clone() for x in prom_memory]

        self.memory.append(state["demand"])
    

    def suggest(self, p):
        first = (self.p - p)/self.diff
        if len(self.memory) > 0:
            second = np.mean(np.array(self.memory).T, axis = 1) / self.C
            # second = torch.stack(self.memory).T.mean(axis = 1) / self.C
        else:
            second = np.ones(self.n)
            # second = torch.ones(self.n)
        
        # res = self.weights[0] * first + self.weights[1] * second
        # e_res = np.exp(res)
        # return self.n * e_res/np.sum(e_res)
        w = torch.sigmoid(self.weight)
        res = w * torch.tensor(first.tolist()) + (1 - w) * torch.tensor(second.tolist())
        e_res = res.exp()
        return self.n * e_res/(e_res).sum()


    def update(self):
        if torch.abs(self.profit_grad) > 0.5:
            prom = (self.weight + self.lr * self.profit_grad).item()
            # print(type(prom), prom)
            self.weight = torch.tensor(prom)
            self.weight.requires_grad = True
        print(self.res, torch.sigmoid(self.weight), self.profit_grad.item())
        # print(self.profit_grad.item())
        self.res = 0
        self.profit_grad = torch.tensor(0.0)


