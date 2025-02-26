### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

### Здесь будут нужные функции для использования в промежуточных частях кода

# Imported relevant python libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# random.seed(42)
# torch.manual_seed(42)

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

        self.previous_memory = None
        self.t = -len(index_list[0])

        if mode == "sanchez_cartas":
            self.beta = 1.5/(10**4)
        elif mode == "zhou":
            self.eps_min = 0.01
            self.eps_max = 1
            self.beta = 1.5/(10**4)

    def __repr__(self):
        return "TQL"

    def suggest(self, memory):
        if type(self.t) != type(None) and self.t < 0:
            idx = np.random.randint(len(self.action_list))
            self.t += 1
            return idx
        
        self.previous_memory = tuple(memory)

        if self.mode == "sanchez_cartas":
            self.eps = np.exp(-self.beta*self.t)
            self.t += 1
        elif self.mode == "zhou":
            self.eps = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.beta*self.t)
            self.t += 1
        
        best = np.argmax(self.Q_mat[tuple(memory)])
        if np.random.random() < self.eps:
            idx = np.random.randint(len(self.action_list))
            while idx == best:
                idx = np.random.randint(len(self.action_list))
            return idx
        else:
            return best

    def update(self, idx, memory, response):
        if self.t == 0:
            self.Q_mat[tuple(memory)][idx] = response

        elif self.t > 0:
            Q = self.Q_mat
            mm = self.previous_memory
            Q[tuple(memory)][idx] = (1 - self.alpha) * Q[tuple(memory)][idx] + self.alpha * (response + self.delta * Q[mm][idx])
            self.Q_mat = Q

        

# print(TQL(0.5,
#     np.zeros((2, 3)),
#     [tuple([0, 0])]*3,
#     [1, 2, 3],
#     0.95,
#     mode = "sanchez_cartas").t)