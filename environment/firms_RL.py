### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requirements.txt

### Здесь будут нужные функции для использования в промежуточных частях кода

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random
import torch.optim as optim
from copy import deepcopy
from collections import deque

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

"""
##################################################
COPYPASTED FOR FUTURE IMPLAMINTATION!
Orginal: https://github.com/00ber/Deep-Q-Networks/blob/main/src/airstriker-genesis/agent.py
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


class DQNet(nn.Module):
    """
    mini cnn structure
    input -> (linear + relu) x 2 -> output
    """

    def __init__(self, input_dim, inventory_actions, price_actions):
        super().__init__()
        
        sloy = 256
        
        self.online = nn.Sequential(
            nn.Linear(input_dim, sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.ReLU(),
        )

        self.inventory_size = len(inventory_actions)
        self.price_size = len(price_actions)

        self.inventory_head = nn.Linear(sloy, self.inventory_size)
        self.price_head = nn.Linear(sloy, self.price_size)
        
        # Target network
        self.target = deepcopy(self.online)
        self.target_inventory = deepcopy(self.inventory_head)
        self.target_price = deepcopy(self.price_head)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
        for p in self.target_inventory.parameters():
            p.requires_grad = False
        for p in self.target_price.parameters():
            p.requires_grad = False

    def forward(self, x, model="online"):
        if model == "online":
            x = self.online(x)
            inv = self.inventory_head(x)
            prc = self.price_head(x)
            return inv, prc
        else:
            x = self.target(x)
            inv = self.target_inventory(x)
            prc = self.target_price(x)
            return inv, prc


class TN_DDQN:
    def __init__(
            self, 
            state_dim: int,
            inventory_actions: list,
            price_actions: list,
            batch_size:int,
            MEMORY_VOLUME: int,
            gamma: float,
            lr: float,
            eps: float,
            mode: str,
            target_update_freq: int,
            memory_size: int,
            cuda_usage = False,
            ):
        
        self.state_dim = state_dim
        self.inventory_actions = inventory_actions
        self.price_actions = price_actions
        self.cuda_usage = cuda_usage

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network
        self.online = DQNet(state_dim, inventory_actions, price_actions)
        self.target = deepcopy(self.online)

        if cuda_usage:
            self.online = DQNet(state_dim, inventory_actions, price_actions).to(self.device)
            self.target = deepcopy(self.online).to(self.device)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.t = - batch_size - MEMORY_VOLUME
        self.mode = mode

        if mode == "sanchez_cartas":
            self.beta = 1.5/(10**4) # 0.005, 1.5/(10**3), 1.5/(10**4)
        elif mode == "zhou":
            self.eps_min = 0.01
            self.eps_max = 1
            self.beta = 1.5/(10**4) # 0.005, 1.5/(10**3)/2, 1.5/(10**3), 1.5/(10**4)
        
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        self.memory = []
        
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)


    def __repr__(self):
        return "TN_DDQN"    


    def _get_state_vector(self, firm_state):
        """Преобразует состояние фирмы в тензор"""
        inventory = firm_state['current_inventory']
        comp_prices = np.array(firm_state['competitors_prices']).flatten()
        if not self.cuda_usage:
            return torch.FloatTensor([inventory] + comp_prices.tolist())
        else:
            return torch.FloatTensor([inventory] + comp_prices.tolist()).to(self.device)


    def suggest_actions(self, firm_state):
        """Возвращает действия (инвентарь, цену)"""
        if self.t < 0:
            inv_idx = random.randint(np.sum(self.inventory_actions < firm_state["current_inventory"]), len(self.inventory_actions)-1)
            price_idx = random.randint(0, len(self.price_actions)-1)
            self.t += 1
            return (inv_idx, price_idx)

        state = self._get_state_vector(firm_state)
        
        if random.random() < self.eps:
            inv_idx = random.randint(np.sum(self.inventory_actions < firm_state["current_inventory"]), len(self.inventory_actions)-1)
            price_idx = random.randint(0, len(self.price_actions)-1)
        else:
            with torch.no_grad():
                inv_q, price_q = self.online(state.unsqueeze(0), 'online')

                if not self.cuda_usage:
                    inventory_actions_tensor = torch.tensor(self.inventory_actions) # , dtype = torch.float16
                else:
                    inventory_actions_tensor = torch.tensor(self.inventory_actions).to(self.device)
                
                mask = inventory_actions_tensor.unsqueeze(0) < state[0]
                inv_q[mask] = -float('inf')

                inv_idx = torch.argmax(inv_q).item()
                price_idx = torch.argmax(price_q).item()
        
        # Decay exploration rate
        if self.mode == "sanchez_cartas":
            self.eps = np.exp(-self.beta*self.t)
        elif self.mode == "zhou":
            self.eps = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.beta*self.t)
        
        self.t += 1

        return (inv_idx, price_idx)


    def cache_experience(self, state, actions, reward, next_state):
        """Сохраняет опыт в replay buffer"""
        state_vec = self._get_state_vector(state)
        next_vec = self._get_state_vector(next_state)

        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        if len(state_vec) == self.state_dim:
            if not self.cuda_usage:
                self.memory.append((
                    state_vec,
                    actions,
                    reward,
                    # torch.LongTensor(rewards),
                    next_vec,
                ))
            else:
                self.memory.append((
                    state_vec.cpu(),  # Храним в CPU памяти
                    actions,
                    torch.FloatTensor(reward).to(self.device),
                    # torch.LongTensor(rewards).to(self.device),
                    next_vec.cpu()
                ))


    def _update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if not self.cuda_usage:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.stack([x[0] for x in batch])
            actions = torch.LongTensor([x[1] for x in batch])
            rewards = torch.FloatTensor([x[2] for x in batch])
            # rewards = torch.stack([x[2] for x in batch])
            next_states = torch.stack([x[3] for x in batch])
        else:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.stack([x[0] for x in batch]).to(self.device)
            actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
            rewards = torch.stack([x[2] for x in batch]).to(self.device)
            next_states = torch.stack([x[3] for x in batch]).to(self.device)

        # Current Q values (online network)
        inv_q_online, price_q_online = self.online(states, 'online')

        if not self.cuda_usage:
            inventory_actions_tensor = torch.tensor(self.inventory_actions)
        else:
            inventory_actions_tensor = torch.tensor(self.inventory_actions).to(self.device)
        
        mask = inventory_actions_tensor.unsqueeze(0) < states[:, 0].unsqueeze(1)
        inv_q_online[mask] = -float('inf')

        inv_selected = inv_q_online.gather(1, actions[:, 0].unsqueeze(1))
        price_selected = price_q_online.gather(1, actions[:, 1].unsqueeze(1))

        # Target Q values (target network)
        with torch.no_grad():
            inv_q_target, price_q_target = self.target(next_states, 'target')
            inv_to_max_online, price_to_max_online = self.online(next_states, 'online')
            
            mask_target = inventory_actions_tensor.unsqueeze(0) < next_states[:, 0].unsqueeze(1)
            inv_q_target[mask_target] = -float('inf')
            inv_to_max_online[mask_target] = -float('inf')
            
            inv_argmax = torch.argmax(inv_to_max_online, dim = 1)
            price_argmax = torch.argmax(price_to_max_online, dim = 1)
            
            inv_max = torch.gather(inv_q_target, 1, inv_argmax.unsqueeze(1)).squeeze()
            price_max = torch.gather(price_q_target, 1, price_argmax.unsqueeze(1)).squeeze()

            targets = rewards + self.gamma * (inv_max + price_max)
            # targets_inv = rewards[:, 0] + self.gamma * (inv_max)
            # targets_price = rewards[:, 1] + self.gamma * (price_max)

        # Loss calculation
        # loss_inv = F.mse_loss(inv_selected.squeeze(), targets)
        # loss_price = F.mse_loss(price_selected.squeeze(), targets)
        ## loss_inv = F.mse_loss(inv_selected.squeeze(), targets_inv)
        ## loss_price = F.mse_loss(price_selected.squeeze(), targets_price)
        # total_loss = loss_inv + loss_price
        
        total_loss = F.mse_loss(inv_selected.squeeze() + price_selected.squeeze(), targets)

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        if self.t >= 0 and self.t % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return total_loss.item()


    def update(self):
        """Обновление модели на основе данных в памяти"""
        loss = self._update_model()
        return loss


    def save(self, path):
        """Сохранение модели"""
        torch.save({
            'model': self.online.state_dict(),
            'exploration': self.eps,
            'memory': self.memory
        }, path)


    def load(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path)
        self.online.load_state_dict(checkpoint['model'])
        self.target.load_state_dict(checkpoint['model'])
        self.eps = checkpoint.get('exploration', 1.0)
        self.memory = checkpoint.get('memory', deque(maxlen=self.memory_size))

