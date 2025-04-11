### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requiremnts.txt
### для сервака: source /mnt/data/venv_new/bin/activate
### для сервака: python3 environment/environment.py

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from scipy.special import expit
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


def calc_demand_grad(x, C, n, doli, boosting,
                     price_val, inv_val, d_grad):
    s_i_s_j = np.prod(boosting)/n
    res = doli * s_i_s_j
    recc = price_val[::-1] - price_val + inv_val - inv_val[::-1]
    K = len(d_grad)
    if K > 0:
        recc += (1 - x)/(C * K) * np.sum(d_grad.T[::-1] - d_grad.T, axis = 1)
    return res * recc


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
            # cuda_usage: bool,
            gamma: float,
            theta_d: float,
            h_plus: float,
            v_minus: float,
			):

        self.weight = starting_weight
        self.demand_memory_size = demand_memory_size
        self.p = p_max
        self.diff = p_max - p_inf
        self.n = n
        assert n == 2, "Right now dynamic platform is built only for 2 players"
        self.C = C

        self.delta = delta
        self.lr = lr
        self.gamma = gamma
        self.theta_d = theta_d
        self.h_plus = h_plus
        self.v_minus = v_minus

        self.memory = []
        self.d_grad = []
        self.res = 0
        self.profit_grad = 0 # torch.tensor(0.0)


    def __repr__(self):
        return "dynamic_weights"


    def cache_data(self, state):
        
        # t = state["timestamp"]
        # if t < self.demand_memory_size and t >= 0:
        #     # print(t)
        #     inv = torch.tensor(state['current_inventory'].tolist())
        #     # prom_memory = [x.clone() for x in self.memory]
        #     # print(self.memory)
        #     boosting = state["boosting"]
        #     demand = torch.tensor(state["demand"])
        #     p = state["competitors_prices"]
        #     res = torch.tensor((self.gamma * p + self.theta_d)) * demand * boosting
        #     res += (self.h_plus * torch.maximum(torch.tensor(0.0), inv - demand))/self.C
        #     res -= (self.v_minus * torch.minimum(torch.tensor(0.0), inv - demand))/self.C
        #     res = self.delta**t * res.sum()
        #     # res = res.sum()
        #     self.res += res.item()
        #     res.backward()
        #     self.profit_grad += self.weight.grad
        #     # self.memory = [x.clone() for x in prom_memory]

        t = state["timestamp"]
        doli = state["doli"]
        boosting = state["boosting"]
        demand = state["demand"]
        price_val = state["price_val"]
        inv_val = state["inv_val"]
        d_grad = calc_demand_grad(expit(self.weight), self.C, self.n, doli,
                                  boosting, price_val, inv_val, np.array(self.d_grad))
        
        if 0 <= t < self.demand_memory_size:
            p = state["competitors_prices"]
            inv = state["current_inventory"]
            # print(d_grad)
            # a += 1
            pi_grad = (self.gamma * p + self.theta_d) * d_grad
            pi_grad = np.sum(pi_grad)
            signs = (np.sign(np.sign(inv - demand) + 0.1) + 1)/2
            signs = (-self.h_plus - self.v_minus) * signs + self.v_minus
            pi_grad += np.sum(signs/self.C * d_grad)
            self.profit_grad += pi_grad * expit(self.weight) * expit(-self.weight)

            # res = (self.gamma * p + self.theta_d) * demand
            # res += (self.h_plus * np.maximum(0, inv - demand))/self.C
            # res -= (self.v_minus * np.minimum(0, inv - demand))/self.C
            res = self.delta**t * state["plat_pi"]
            # res = res.sum()
            self.res += res

        self.memory.append(demand)
        self.d_grad.append(d_grad)

        if len(self.memory) > self.demand_memory_size:
            self.memory.pop(0)
            self.d_grad.pop(0)
    

    def suggest(self, p):
        first = (self.p - p)/self.diff
        if len(self.memory) > 0:
            second = np.mean(np.array(self.memory).T, axis = 1) / self.C
            # second = torch.stack(self.memory).T.mean(axis = 1) / self.C
        else:
            second = np.ones(self.n)
            # second = torch.ones(self.n)
        
        res = expit(self.weight) * first + (1 - expit(self.weight)) * second
        e_res = np.exp(res)
        return (first, second, self.n * e_res/np.sum(e_res))
    
        # w = torch.sigmoid(self.weight)
        # res = w * torch.tensor(first.tolist()) + (1 - w) * torch.tensor(second.tolist())
        # e_res = res.exp()
        # return self.n * e_res/(e_res).sum()


    def update(self):
        # if torch.abs(self.profit_grad) > 0.5:
        #     prom = (self.weight + self.lr * self.profit_grad).item()
        #     # print(type(prom), prom)
        #     self.weight = torch.tensor(prom)
        #     self.weight.requires_grad = True
        # print(self.res, torch.sigmoid(self.weight), self.profit_grad.item())
        # # print(self.profit_grad.item())
        # self.res = 0
        # self.profit_grad = torch.tensor(0.0)

        if np.abs(self.profit_grad) > 0.15:
            # print(self.res, expit(self.weight), self.profit_grad)
            self.weight = self.weight + self.lr * self.profit_grad/self.demand_memory_size
            # self.d_grad = []
        
        # print(self.res, expit(self.weight), self.profit_grad)
        self.res = 0
        self.profit_grad = 0


