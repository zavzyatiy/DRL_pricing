### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from firms.firms import epsilon_greedy

### иницализация randomseed

random.seed(42)
torch.manual_seed(42)

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

### количество итераций внутри среды
T = 100000

### количество симуляций среды
ENV = 1

n = 2               # число фирм
m = 5               # горизонт памяти платформы
delta = 0.6         # коэффициент дисконтирования
gamma = 0.5         # склонность доверия рекламе
c_i = 1             # издержки закупки товара
h_plus = 0          # издержки хранения остатков
v_minus = 0         # издержки экстренного дозаполнения
eta = 0.05          # вероятность оставить отзыв
p_inf = c_i         # нижняя граница цены == MC
p_sup = 2.2     # верхняя граница цены == адеватность/монополия
arms_amo = 100      # количество цен для перебора при дискретизации
                    # пространства возможных цен
mode = "D"          # режим: D - дискр., C - непр.

if mode == "D":
    prices = np.linspace(p_inf, p_sup, arms_amo + 1)
else:
    prices = (p_inf, p_sup)

eps = 0.6

history1 = []
history2 = []
History1 = []
History2 = []

### Всевозможные модели спросов

class demand_function:
     
    def __init__(
            self,
            n: int,
            mode: str,           
			):

        mode_list = ["logit"]
        assert n >= 2, "Demand is built for olygopoly, so n > 2!"
        assert mode in mode_list, f"Demand function must be in [{' '.join(mode_list)}]"
        self.n = n
        self.mode = mode
    
    def distribution(self, prices):
        if self.mode == "logit":
            s = [0] + list(prices)
            exp_s = [np.exp(-x) for x in s]
            sum_exp = sum(exp_s)
            return [x/sum_exp for x in exp_s[1:]]

### ПОКА ВСЕ НАПИСАНО ДЛЯ "D"
### Более того, для эпсилон-жадной реализации QL без памяти
for env in range(ENV):

    spros = demand_function(n, "logit")

    ### Инициализация алгоритмов фирм
    # FirmMemory = ... 
    firm1 = epsilon_greedy(eps,
                           np.zeros(len(prices)),
                           prices,
                           mode = "sanchez_cartas",
                           )
    
    firm2 = epsilon_greedy(eps,
                           np.zeros(len(prices)),
                           prices,
                           mode = "sanchez_cartas",
                           )

    ### Инициализация памяти платформы
    # -
    
    ### Инициализация памяти в отзывах
    # -

    ### Инициализация основного цикла
    for t in tqdm(range(T)):

        ### действие
        p1 = firm1.suggest()
        p2 = firm2.suggest()

        ### подсчет спроса
        doli = spros.distribution([p1, p2])

        ### подсчет прибыли фирм
        pi_1 = (p1 - c_i) * doli[0]
        pi_2 = (p2 - c_i) * doli[1]

        History1.append(pi_1)
        History2.append(pi_2)

        ### обновление весов алгоритмов
        firm1.update(p1, pi_1)
        firm2.update(p2, pi_2)

        history1.append(p1)
        history2.append(p2)

# plt.plot(history1)
# plt.plot(history2)
# plt.show()

# plt.plot(History1)
# plt.plot(History2)
# plt.show()

window_size = 100
kernel = np.ones(window_size) / window_size
mv1 = np.convolve(history1, kernel, mode='valid')
mv2 = np.convolve(history2, kernel, mode='valid')

plt.plot(mv1)
plt.plot(mv2)
plt.show()

window_size = 100
kernel = np.ones(window_size) / window_size
mv1 = np.convolve(History1, kernel, mode='valid')
mv2 = np.convolve(History2, kernel, mode='valid')

plt.plot(mv1)
plt.plot(mv2)
plt.show()