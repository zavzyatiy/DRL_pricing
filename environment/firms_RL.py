### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

### Здесь будут нужные функции для использования в промежуточных частях кода

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random
from copy import deepcopy
from collections import deque

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

"""
####################################################################################################
Code for TN_DDQN is a custom version of 00ber implementation of DQN.
Orginal: https://github.com/00ber/Deep-Q-Networks/blob/main/src/airstriker-genesis/agent.py
Also important for invalid action masking: https://github.com/vwxyzjn/invalid-action-masking/blob/master/ppo.py
####################################################################################################
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class epsilon_greedy:
     
    def __init__(
            self,
            eps: float,
            Q_list: list,
            action_list: list,
            alpha = 0.5,
            mode = None,
			):

        mode_list = ["sanchez_cartas", "zhou"]
        assert len(Q_list) == len(action_list), "Length doesn't match!"
        assert type(mode) == type(None) or mode in mode_list, f"Search mode must be in [None {' '.join(mode_list)}]"

        self.eps = eps
        self.Q_list = Q_list
        self.action_list = action_list
        self.memory = [0] * len(action_list)
        self.alpha = alpha
        self.mode = mode

        if mode == "sanchez_cartas":
            self.t = 0
            self.beta = 1.5/(10**4)
        elif mode == "zhou":
            self.t = 0
            self.eps_min = 0.05
            self.eps_max = 1
            self.beta = 1.5/(10**4)

    def __repr__(self):
        return "epsilon_greedy"

    def suggest(self):
        if self.mode == "sanchez_cartas":
            self.eps = np.exp(-self.beta*self.t)
            self.t += 1
        elif self.mode == "zhou":
            self.eps = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.beta*self.t)
            self.t += 1
        
        best = np.argmax(self.Q_list)
        if np.random.random() < self.eps:
            idx = np.random.randint(len(self.action_list))
            while idx == best:
                idx = np.random.randint(len(self.action_list))
            self.memory[idx] += 1
            return idx
        else:
            return best

    def update(self, idx, response):
        Q_list = self.Q_list
        
        if self.mode:
            Q_list[idx] = ((self.memory[idx] - 1) * Q_list[idx] + response)/self.memory[idx]
        else:
            if self.memory[idx] == 1:
                Q_list[idx] = response
            else:
                Q_list[idx] = self.alpha * Q_list[idx] + (1 - self.alpha) * response
        
        self.Q_list = Q_list
    

class TQL:
     
    def __init__(
            self,
            eps: float,
            Q_mat: list,
            MEMORY_VOLUME: int,
            n: int,
            own: bool,
            ONLY_OWN: bool, 
            index_list: list,
            action_list: list,
            delta: float,
            alpha = 0.5,
            mode = None,
			):

        mode_list = ["sanchez_cartas", "zhou"]
        assert Q_mat.shape[1] == len(action_list), "Length doesn't match!"
        assert type(mode) == type(None) or mode in mode_list, f"Search mode must be in [None {' '.join(mode_list)}]"

        self.eps = eps
        self.Q_mat = Q_mat
        self.index_list = index_list
        self.action_list = action_list
        self.alpha = alpha
        self.delta = delta
        self.mode = mode
        self.MEMORY_VOLUME = MEMORY_VOLUME
        self.n = n
        self.own = 1 - int(own)
        if ONLY_OWN:
            self.n = 1
            self.own = 0

        self.previous_memory = None
        self.t = - MEMORY_VOLUME

        if mode == "sanchez_cartas":
            self.beta = 1.5/(10**(4 + 0* MEMORY_VOLUME*(n + self.own)))
        elif mode == "zhou":
            self.eps_min = 0.025
            self.eps_max = 1
            self.beta = 1.5/(10**(4 + 0* MEMORY_VOLUME*(n + self.own)))

    def __repr__(self):
        return "TQL"

    def adjust_memory(self, memory):
        ### преобразование исхода в индекс для хранения
        if self.t >= 0:
            MV = self.MEMORY_VOLUME
            L = len(self.action_list)
            n = self.n
            own = self.own
            syst_prorm = L
            syst_mem = L**(n - own)
            prom = [sum([x[i] * syst_prorm**(n - own - 1 - i) for i in range(n - own)]) for x in memory]
            mem = sum([prom[i] * syst_mem**(MV - 1 - i) for i in range(MV)])
            self.previous_memory = mem

    def suggest(self):
        if self.t < 0:
            idx = np.random.randint(len(self.action_list))
            self.t += 1
            return idx
        
        mem = self.previous_memory

        if self.mode == "sanchez_cartas":
            self.eps = np.exp(-self.beta*self.t)
        elif self.mode == "zhou":
            self.eps = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.beta*self.t)
        
        self.t += 1
        
        best = np.argmax(self.Q_mat[mem])
        if np.random.random() < self.eps:
            idx = np.random.randint(len(self.action_list))
            while idx == best:
                idx = np.random.randint(len(self.action_list))
            return idx
        else:
            return best

    def update(self, idx, learn, response):
        
        if self.t >= 1:
            MV = self.MEMORY_VOLUME
            L = len(self.action_list)
            n = self.n
            own = self.own
            syst_prorm = L
            syst_mem = L**(n - own)
            prom = [sum([x[i] * syst_prorm**(n - own - 1 - i) for i in range(n - own)]) for x in learn]
            lr = sum([prom[i] * syst_mem**(MV - 1 - i) for i in range(MV)])

            Q = self.Q_mat
            mm = self.previous_memory
            # print("Память фирмы", mm)
            Q[mm, idx] = (1 - self.alpha) * Q[mm, idx] + self.alpha * (response + self.delta * np.max(Q[lr]))
            self.Q_mat = Q





class DQN(nn.Module):
    """
    mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, action_dim):

        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        
        self.target = deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class TN_DDQN:

    def __init__(
            self,
			):

        pass

    def __repr__(self):
        return "TN_DDQN"

    def suggest(self, memory):
        pass

    def update(self, idx, memory, response):
        pass



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