### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random as rnd
import numpy as np
import torch

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from firms.firms import epsilon_greedy

### иницализация randomseed

rnd.seed(42)
torch.manual_seed(42)

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

### количество итераций внутри среды
T = 100

### количество симуляций среды
ENV = 2

n = 2               # число фирм
m = 5               # горизонт памяти платформы
delta = 0.6         # коэффициент дисконтирования
gamma = 0.5         # склонность доверия рекламе
c_i = 1             # издержки закупки товара
h_plus = 0          # издержки хранения остатков
v_minus = 0         # издержки экстренного дозаполнения
eta = 0.05          # вероятность оставить отзыв
p_inf = c_i         # нижняя граница цены == MC
p_sup = 10          # верхняя граница цены == адеватность/монополия
arms_amo = 100      # количество цен для перебора при дискретизации
                    # пространства возможных цен
mode = "D"          # режим: D - дискр., C - непр.

if mode == "D":
    prices = np.linspace(p_inf, p_sup, arms_amo + 1)
else:
    prices = (p_inf, p_sup)

eps = 0.8
# ex = epsilon_greedy(0.5, [], [])
# ex.hello()

### ПОКА ВСЕ НАПИСАНО ДЛЯ "D"
for env in range(ENV):

    ### Инициализация алгоритмов фирм
    # FirmMemory = ... 
    firm1 = epsilon_greedy(eps,
                           np.random.normal(0, 1, size=arms_amo + 1),
                           prices)

    ### Инициализация памяти платформы

    ### Инициализация памяти в отзывах

    ### Инициализация основного цикла
    for t in range(T):

        ### взаимодействие
        

        continue
