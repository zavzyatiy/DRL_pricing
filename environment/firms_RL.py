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
##################################################
Code for TN_DDQN is a custom version of 00ber implementation of DQN.
Orginal: https://github.com/00ber/Deep-Q-Networks/blob/main/src/airstriker-genesis/agent.py
Also important for invalid action masking: https://github.com/vwxyzjn/invalid-action-masking/blob/master/ppo.py
##################################################
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




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CategoricalMasked(Categorical):
    def __init__(self, logits, masks):
        self.masks = masks.bool().to(device)
        logits = torch.where(self.masks, logits, torch.tensor(-1e8, device=device))
        super().__init__(logits=logits)


class CustomNet(nn.Module):
    def __init__(self, input_dim, n_stock_actions, n_price_actions):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.head_stock = nn.Linear(256, n_stock_actions)  # Логиты для выбора запаса
        self.head_price = nn.Linear(256, n_price_actions)  # Логиты для выбора цены

    def forward(self, x):
        x = self.shared_net(x)
        stock_logits = self.head_stock(x)
        price_logits = self.head_price(x)
        return stock_logits, price_logits


class InventoryAgent:


    def __init__(
        self,
        state_dim,
        stock_actions,
        price_actions,
        c_i,
        p_max,
        gamma=0.9,
        batch_size=64,
        memory_size=10000
    ):
        self.stock_actions = np.array(stock_actions)  # Допустимые уровни запасов
        self.price_actions = np.array(price_actions)  # Допустимые цены
        self.c_i = c_i
        self.p_max = p_max
        self.gamma = gamma
        self.batch_size = batch_size

        # Инициализация нейросетей
        self.net = CustomNet(state_dim, len(stock_actions), len(price_actions)).to(device)
        self.target_net = deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        # Replay buffer
        self.memory = deque(maxlen=memory_size)


    def get_masks(self, current_stock):
        """Генерация масок для запасов и цен."""
        # Маска для запасов: только значения > current_stock
        stock_mask = torch.tensor(self.stock_actions > current_stock, device=device)
        
        # Маска для цен: значения в [c_i, p_max]
        price_mask = torch.tensor(
            (self.price_actions >= self.c_i) & (self.price_actions <= self.p_max),
            device=device
        )
        return stock_mask, price_mask


    def act(self, state, current_stock):
        """Выбор действия с учетом масок."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            stock_logits, price_logits = self.net(state_tensor)
            stock_mask, price_mask = self.get_masks(current_stock)
            
            # Применение масок
            stock_dist = CategoricalMasked(stock_logits, stock_mask)
            price_dist = CategoricalMasked(price_logits, price_mask)
            
            stock_idx = stock_dist.sample().item()
            price_idx = price_dist.sample().item()

        return (
            self.stock_actions[stock_idx],  # Выбранный запас
            self.price_actions[price_idx]    # Выбранная цена
        )


    def cache(self, state, action_stock, action_price, reward, next_state):
        """Сохранение опыта в replay buffer"""
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        self.memory.append((state, action_stock, action_price, reward, next_state))


    def learn(self):
        """Обновление нейросети на основе replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        # Формирование батча
        batch = random.sample(self.memory, self.batch_size)
        states, stock_acts, price_acts, rewards, next_states = zip(*batch)
        
        # Конвертация в тензоры
        states = torch.stack(states).to(device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        
        # Q-значения для текущих действий
        stock_logits, price_logits = self.net(states)
        stock_q = stock_logits.gather(1, torch.LongTensor(stock_acts).view(-1, 1).to(device))
        price_q = price_logits.gather(1, torch.LongTensor(price_acts).view(-1, 1).to(device))
        q_values = stock_q + price_q

        # Целевые Q-значения
        with torch.no_grad():
            next_stock_logits, next_price_logits = self.target_net(next_states)
            next_q = next_stock_logits.max(1)[0] + next_price_logits.max(1)[0]
            target_q = rewards + self.gamma * next_q

        # Обучение
        loss = torch.nn.functional.mse_loss(q_values, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_net(self):
        """Синхронизация целевой сети."""
        self.target_net.load_state_dict(self.net.state_dict())



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

# Допустимые действия
stock_actions = [50, 60, 70, 80]  # Уровни запасов
price_actions = [10, 20, 30, 40]  # Цены
# stock_actions = np.array(stock_actions)/100


# Инициализация агента
agent = InventoryAgent(
    state_dim=4,  # Размер состояния (например: текущий запас, последние 3 цены)
    stock_actions=stock_actions,
    price_actions=price_actions,
    c_i=15,       # Минимальная цена
    p_max=40,
    gamma=0.95
)

# Пример взаимодействия со средой
state = [70, 20, 25, 30]  # Текущее состояние: запас=70, последние цены=[20,25,30]
current_stock = 70

# Выбор действия
next_stock, next_price = agent.act(state, current_stock)
print(f"Выбрано: запас={next_stock}, цена={next_price}")  # Например: 80, 20

# Обучение агента
agent.cache(state, next_stock, next_price, reward=10.5, next_state=[next_stock, 20, 25, 30])
agent.learn()
agent.update_target_net()

print(device)