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

from firms.firms import epsilon_greedy, TQL

### иницализация randomseed

# random.seed(42)
# torch.manual_seed(42)

### начальные условия: инициализация весов для всех алгоритмов
### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

### количество итераций внутри среды
T = 100000

### количество симуляций среды
ENV = 1

# число фирм
n = 2
# горизонт памяти платформы
m = 5
# коэффициент дисконтирования
delta = 0.95
# склонность доверия рекламе
gamma = 0.5
# издержки закупки товара
c_i = 0
# издержки хранения остатков
h_plus = 0
# издержки экстренного дозаполнения
v_minus = 0
# вероятность оставить отзыв
eta = 0.05
# нижняя граница цены == MC
p_inf = c_i
# верхняя граница цены == адеватность/монополия
p_sup = 2
# количество цен для перебора при дискретизации
# пространства возможных цен
arms_amo = 101
# режим: D - дискр., C - непр.
mode = "D"

if mode == "D":
    prices = np.linspace(p_inf, p_sup, arms_amo)
else:
    prices = (p_inf, p_sup)

### параметры для тестирования
eps = 0.6
a = 2
mu = 0.25
alpha = 0.15

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
        if self.mode == "logit":
            s = list(prices)
            if self.a:
                exp_s = [1] + [np.exp((self.a - x)/self.mu) for x in s]
            else:
                exp_s = [1] + [np.exp(-x) for x in s]
            sum_exp = sum(exp_s)
            return [x/sum_exp for x in exp_s[1:]]

### ПОКА ВСЕ НАПИСАНО ДЛЯ "D"
### Более того, для эпсилон-жадной реализации QL без памяти
for env in range(ENV):

    spros = demand_function(n, "logit", a = a, mu = mu)

    ### Инициализация алгоритмов фирм
    # FirmMemory = ... 
    firm1 = epsilon_greedy(eps,
                           np.zeros(len(prices)),
                           prices,
                        #    mode= "sanchez_cartas",
                           mode= "zhou",
                           )
    
    firm2 = epsilon_greedy(eps,
                           np.zeros(len(prices)),
                           prices,
                        #    mode= "sanchez_cartas"
                           mode= "zhou",

                           )

    # firm1 = TQL()

    # firm2 = TQL()

    ### Инициализация памяти платформы
    # -
    
    ### Инициализация памяти в отзывах
    # -

    ### Инициализация основного цикла
    for t in tqdm(range(T)):

        ### действие
        idx1 = firm1.suggest()
        idx2 = firm2.suggest()
        p1 = prices[idx1]
        p2 = prices[idx2]

        ### подсчет спроса
        doli = spros.distribution([p1, p2])

        ### подсчет прибыли фирм
        pi_1 = (p1 - c_i) * doli[0]
        pi_2 = (p2 - c_i) * doli[1]

        History1.append(pi_1)
        History2.append(pi_2)

        ### обновление весов алгоритмов
        firm1.update(idx1, pi_1)
        firm2.update(idx2, pi_2)

        history1.append(p1)
        history2.append(p2)

# plt.plot(history1)
# plt.plot(history2)
# plt.show()

# plt.plot(History1)
# plt.plot(History2)
# plt.show()

window_size = int(T/20)
kernel = np.ones(window_size) / window_size
mv1 = np.convolve(history1, kernel, mode='valid')
mv2 = np.convolve(history2, kernel, mode='valid')

plt.plot(mv1)
plt.plot(mv2)
plt.show()

window_size = int(T/20)
kernel = np.ones(window_size) / window_size
mv1 = np.convolve(History1, kernel, mode='valid')
mv2 = np.convolve(History2, kernel, mode='valid')

plt.plot(mv1)
plt.plot(mv2)
plt.show()