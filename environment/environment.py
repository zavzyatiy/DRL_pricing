### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random as rnd
import numpy as np
import torch

### иницализация randomseed

rnd.seed(42)
torch.manual_seed(42)

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

T = 100
n = 2
m = 5
delta = 0.6
gamma = 0.5
c_i = 1
h_plus = 0
v_minus = 0
eta = 0.05

### Инициализация памяти фирм

### Инициализация памяти платформы

### Инициализация памяти в отзывах

### Инициализация основного цикла

for t in range(T):

    ### взаимодействие

    continue
