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
from copy import deepcopy

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


"""
##################################################
Projects for ideas and code:
TN-DDQN: https://github.com/00ber/Deep-Q-Networks/blob/main/src/airstriker-genesis/agent.py , DeepSeek R1
PPO: https://github.com/vwxyzjn/invalid-action-masking , https://github.com/Larry-Liu02/Dynamic-Pricing-Algorithm/ , DeepSeek R1
SAC: https://github.com/denisyarats/pytorch_sac , https://github.com/Larry-Liu02/Dynamic-Pricing-Algorithm/ , DeepSeek R1
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
            cuda_usage: bool,
            eps_min: float,
            eps_max: float,
            beta: float,
            dtype = torch.float32,
            ):
        
        self.state_dim = state_dim
        self.inventory_actions = inventory_actions
        self.price_actions = price_actions
        self.cuda_usage = cuda_usage

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dtype = dtype
        self.inventory_actions_tensor = torch.tensor(self.inventory_actions, dtype = self.dtype).to(self.device)

        # Q-network
        self.online = DQNet(state_dim, inventory_actions, price_actions).to(self.device)
        self.target = deepcopy(self.online).to(self.device)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.t = - batch_size - MEMORY_VOLUME
        self.mode = mode

        self.eps_min = eps_min # 0.01
        self.eps_max = eps_max # 1
        self.beta = beta # 0.005, 1.5/(10**3)/2, 1.5/(10**3), 1.5/(10**4)
        
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
        return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)


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
                
                mask = self.inventory_actions_tensor.unsqueeze(0) < state[0]
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
                    next_vec,
                ))
            else:
                self.memory.append((
                    state_vec.cpu(),  # Храним в CPU памяти
                    actions,
                    torch.tensor([reward], dtype = self.dtype).to(self.device),
                    next_vec.cpu()
                ))


    def _update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if not self.cuda_usage:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.stack([x[0] for x in batch])
            actions = torch.LongTensor([x[1] for x in batch])
            rewards = torch.tensor([x[2] for x in batch], dtype = self.dtype)
            next_states = torch.stack([x[3] for x in batch])
        else:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.stack([x[0] for x in batch]).to(self.device)
            actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
            rewards = torch.stack([x[2] for x in batch], dim = 1)[0].to(self.device)
            next_states = torch.stack([x[3] for x in batch]).to(self.device)

        # Current Q values (online network)
        inv_q_online, price_q_online = self.online(states, 'online')

        mask = self.inventory_actions_tensor.unsqueeze(0) < states[:, 0].unsqueeze(1)
        inv_q_online[mask] = -float('inf')

        inv_selected = inv_q_online.gather(1, actions[:, 0].unsqueeze(1))
        price_selected = price_q_online.gather(1, actions[:, 1].unsqueeze(1))

        # Target Q values (target network)
        with torch.no_grad():
            inv_q_target, price_q_target = self.target(next_states, 'target')
            inv_to_max_online, price_to_max_online = self.online(next_states, 'online')
            
            mask_target = self.inventory_actions_tensor.unsqueeze(0) < next_states[:, 0].unsqueeze(1)
            inv_q_target[mask_target] = -float('inf')
            inv_to_max_online[mask_target] = -float('inf')
            
            inv_argmax = torch.argmax(inv_to_max_online, dim = 1)
            price_argmax = torch.argmax(price_to_max_online, dim = 1)

            inv_max = torch.gather(inv_q_target, 1, inv_argmax.unsqueeze(1)).squeeze()
            price_max = torch.gather(price_q_target, 1, price_argmax.unsqueeze(1)).squeeze()

            targets = rewards + self.gamma * (inv_max + price_max)

        # Loss calculation
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


class PPO_D_ActorNet(nn.Module):
    def __init__(self, input_dim, inventory_actions, price_actions):
        super().__init__()
        
        sloy = 256 # 256
        
        self.d_actor_net = nn.Sequential(
            nn.Linear(input_dim, sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.ReLU(),
        )

        self.inventory_size = len(inventory_actions)
        self.price_size = len(price_actions)

        self.inventory_head = nn.Linear(sloy, self.inventory_size)
        self.price_head = nn.Linear(sloy, self.price_size)
        

    def forward(self, x):
        x = self.d_actor_net(x)
        raw_inv = self.inventory_head(x)
        raw_prc = self.price_head(x)
        return raw_inv, raw_prc


class PPO_D_CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.d_critic_net = nn.Sequential(
            nn.Linear(input_dim, sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.ReLU(),
            nn.Linear(sloy, 1)
            )


    def forward(self, x):
        return self.d_critic_net(x)


def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)


class PPO_D:
    def __init__(
            self, 
            state_dim: int,
            inventory_actions: list,
            price_actions: list,
            batch_size: int,
            N_epochs: int,
            gamma: float,
            actor_lr: float,
            critic_lr: float,
            epochs: int,
            clip_eps: float,
            lmbda: float,
            cuda_usage: bool,
            dtype = torch.float32,
            ):
        
        self.state_dim = state_dim
        self.inventory_actions = inventory_actions
        self.price_actions = price_actions
        self.cuda_usage = cuda_usage

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dtype = dtype
        self.inventory_actions_tensor = torch.tensor(self.inventory_actions, dtype = self.dtype).to(self.device)

        # Actor and Critic networks
        self.d_actor_net = PPO_D_ActorNet(state_dim, inventory_actions, price_actions).to(self.device)
        self.d_critic_net = PPO_D_CriticNet(state_dim).to(self.device)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.memory_size = N_epochs
        self.lmbda = lmbda
        self.memory = []
        
        self.actor_optimizer = torch.optim.Adam(self.d_actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.d_critic_net.parameters(), lr=critic_lr)


    def __repr__(self):
        return "PPO_D"


    def _get_state_vector(self, firm_state):
        """Преобразует состояние фирмы в тензор"""
        if not self.cuda_usage:
            inventory = firm_state['current_inventory']
            comp_prices = np.array(firm_state['competitors_prices']).flatten()
            return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)
        else:
            inventory = firm_state['current_inventory']
            comp_prices = np.array(firm_state['competitors_prices']).flatten()
            return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)


    def suggest_actions(self, firm_state):
        """Возвращает действия (инвентарь, цену)"""
        state = self._get_state_vector(firm_state)
        
        raw_inv, raw_prc = self.d_actor_net(state)
        price_prob = torch.softmax(raw_prc, dim = 0)
        mask = self.inventory_actions_tensor.unsqueeze(0) < state[0]

        raw_inv[mask[0]] = -1e8
        inv_prob = torch.softmax(raw_inv, dim = 0)

        # print("Распред. Inv", inv_prob)
        # print("Распред. Price", price_prob)

        sampled_inv_index = torch.multinomial(inv_prob, num_samples=1)
        sampled_prc_index = torch.multinomial(price_prob, num_samples=1)

        # print("Вероятность выбора объема:", inv_prob[sampled_inv_index])

        return (sampled_inv_index.item(), sampled_prc_index.item())


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
                    next_vec,
                ))
            else:
                self.memory.append((
                    state_vec.cpu(),
                    actions,
                    torch.tensor([reward], dtype = self.dtype).to(self.device),
                    next_vec.cpu(),
                ))


    def update(self):
        """Обновление модели на основе данных в памяти"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if not self.cuda_usage:
            all_states = torch.stack([x[0] for x in self.memory])
            all_actions = torch.LongTensor([x[1] for x in self.memory])
            all_rewards = torch.tensor([x[2] for x in self.memory], dtype = self.dtype).view(-1, 1)
            all_next_states = torch.stack([x[3] for x in self.memory])
        else:
            all_states = torch.stack([x[0] for x in self.memory]).to(self.device)
            all_actions = torch.LongTensor([x[1] for x in self.memory]).to(self.device)
            all_rewards = torch.stack([x[2] for x in self.memory], dim = 1)[0].view(-1, 1).to(self.device)
            all_next_states = torch.stack([x[3] for x in self.memory]).to(self.device)
        
        td_target = all_rewards + self.gamma * self.d_critic_net(all_next_states)
        td_delta = td_target - self.d_critic_net(all_states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # print("all_rewards", all_rewards)
        # print(td_delta)

        with torch.no_grad():
            all_raw_inv, all_raw_prc = self.d_actor_net(all_states)
            mask_ = self.inventory_actions_tensor.unsqueeze(0) < all_states[:, 0].unsqueeze(1)
            all_raw_inv[mask_] = -1e8

            inv_prob = torch.softmax(all_raw_inv, dim = 1)
            price_prob = torch.softmax(all_raw_prc, dim = 1)

            old_inv_probs = inv_prob.gather(1, all_actions[:, 0].unsqueeze(1))
            # print("Какая используется вероятность выбора объема", old_inv_probs)
            old_price_probs = price_prob.gather(1, all_actions[:, 1].unsqueeze(1))

        index_list = [i for i in range(len(self.memory))]

        for _ in range(self.epochs):
            # print("Итерация обновления", _)
            batch = random.sample(index_list, self.batch_size)
            states = all_states[batch]
            actions = all_actions[batch]
            local_advantage = advantage[batch]
            
            raw_inv, raw_prc = self.d_actor_net(states)
            mask = self.inventory_actions_tensor.unsqueeze(0) < states[:, 0].unsqueeze(1)
            raw_inv[mask] = -1e8

            Prob_inv = torch.softmax(raw_inv, dim = 1)
            Prob_price = torch.softmax(raw_prc, dim = 1)

            inv_probs = Prob_inv.gather(1, actions[:, 0].unsqueeze(1))
            price_probs = Prob_price.gather(1, actions[:, 1].unsqueeze(1))

            ratio = (inv_probs * price_probs)/(old_inv_probs[batch] * old_price_probs[batch])
            surr1 = ratio * local_advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * local_advantage
            
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = F.mse_loss(self.d_critic_net(states), td_target[batch].detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # print("#"*40)
            

class PPO_C_ActorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.c_actor_net = nn.Sequential(
            nn.Linear(input_dim, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
        )

        self.inventory_head_mu = nn.Linear(sloy, 1)
        self.inventory_head_sigma = nn.Linear(sloy, 1)
        self.price_head_mu = nn.Linear(sloy, 1)
        self.price_head_sigma = nn.Linear(sloy, 1)
        

    def forward(self, x):
        y = self.c_actor_net(x)
        inv_mu, inv_sigma = self.inventory_head_mu(y), self.inventory_head_sigma(y)
        prc_mu, prc_sigma = self.price_head_mu(y), self.price_head_sigma(y)
        assert not torch.isnan(inv_mu).any(), "inv_mu is NaN"
        assert not torch.isnan(prc_mu).any(), "prc_mu is NaN"
        # inv_sigma, prc_sigma = torch.exp(log_inv_sigma).clip(0.1, 5), torch.exp(log_prc_sigma).clip(0.1, 5)
        inv_sigma = torch.pow(1 + torch.pow(inv_sigma, 2), 0.5) - 1
        inv_sigma = inv_sigma.clamp(min = 0.01)
        prc_sigma = torch.pow(1 + torch.pow(prc_sigma, 2), 0.5) - 1
        prc_sigma = prc_sigma.clamp(min = 0.01)
        return (inv_mu, inv_sigma, prc_mu, prc_sigma)


class PPO_C_CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.c_critic_net = nn.Sequential(
            nn.Linear(input_dim, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, 1)
            )


    def forward(self, x):
        return self.c_critic_net(x)


class PPO_C:
    def __init__(
            self, 
            state_dim: int,
            inventory_actions: list,
            price_actions: list,
            batch_size: int,
            N_epochs: int,
            gamma: float,
            actor_lr: float,
            critic_lr: float,
            epochs: int,
            clip_eps: float,
            lmbda: float,
            cuda_usage: bool,
            dtype = torch.float32,
            ):
        
        self.state_dim = state_dim
        self.inventory_actions = inventory_actions
        self.price_actions = price_actions
        self.cuda_usage = cuda_usage

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dtype = dtype
        self.price_diff = torch.tensor(self.price_actions[1] - self.price_actions[0])

        # Actor and Critic networks
        self.c_actor_net = PPO_C_ActorNet(state_dim).to(self.device)
        self.c_critic_net = PPO_C_CriticNet(state_dim).to(self.device)
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.memory_size = N_epochs
        self.lmbda = lmbda
        self.memory = []
        
        self.actor_optimizer = torch.optim.Adam(self.c_actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.c_critic_net.parameters(), lr=critic_lr)


    def __repr__(self):
        return "PPO_C"


    def _get_state_vector(self, firm_state):
        """Преобразует состояние фирмы в тензор"""
        if not self.cuda_usage:
            inventory = firm_state['current_inventory']
            comp_prices = np.array(firm_state['competitors_prices']).flatten()
            return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)
        else:
            inventory = firm_state['current_inventory']
            comp_prices = np.array(firm_state['competitors_prices']).flatten()
            return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)


    def suggest_actions(self, firm_state):
        """Возвращает действия (инвентарь, цену)"""
        state = self._get_state_vector(firm_state)
        
        inv_mu, inv_sigma, prc_mu, prc_sigma = self.c_actor_net(state)
        
        u_inv = torch.distributions.Normal(inv_mu, inv_sigma).sample()
        u_prc = torch.distributions.Normal(prc_mu, prc_sigma).sample()
        inv = state[0] + torch.sigmoid(u_inv/10).clamp(1e-4, 1-1e-4) * (self.inventory_actions[1] - state[0])
        price = self.price_actions[0] + torch.sigmoid(u_prc/10).clamp(1e-4, 1-1e-4) * (self.price_actions[1] - self.price_actions[0])

        return (inv.item(), price.item(), u_inv.item(), u_prc.item())


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
                    next_vec,
                ))
            else:
                self.memory.append((
                    state_vec.cpu(),
                    torch.tensor(actions, dtype = self.dtype).to(self.device),
                    torch.tensor([reward], dtype = self.dtype).to(self.device),
                    next_vec.cpu(),
                ))


    def update(self):
        """Обновление модели на основе данных в памяти"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if not self.cuda_usage:
            all_states = torch.stack([x[0] for x in self.memory])
            all_actions = torch.tensor([x[1] for x in self.memory])
            all_rewards = torch.tensor([x[2] for x in self.memory], dtype = self.dtype).view(-1, 1)
            all_next_states = torch.stack([x[3] for x in self.memory])
        else:
            all_states = torch.stack([x[0] for x in self.memory]).to(self.device)
            all_actions = torch.stack([x[1] for x in self.memory]).to(self.device)
            all_rewards = torch.stack([x[2] for x in self.memory], dim = 1)[0].view(-1, 1).to(self.device)
            all_next_states = torch.stack([x[3] for x in self.memory]).to(self.device)
        
        td_target = all_rewards + self.gamma * self.c_critic_net(all_next_states)
        td_delta = td_target - self.c_critic_net(all_states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # print("all_rewards", all_rewards)
        # print(td_delta)

        with torch.no_grad():
            inv_mu, inv_sigma, prc_mu, prc_sigma = self.c_actor_net(all_states)
            inv_dist = torch.distributions.Normal(inv_mu.detach(), inv_sigma.detach())
            price_dist = torch.distributions.Normal(prc_mu.detach(), prc_sigma.detach())

            # print("all_actions", all_actions)
            # print("all_actions[:, 0]", all_actions[:, 0])

            old_inv_ln_probs = inv_dist.log_prob(all_actions[:, 0]).diagonal()
            # old_inv_ln_probs += F.logsigmoid(all_actions[:, 0])
            # old_inv_ln_probs += F.logsigmoid(-all_actions[:, 0])
            # old_inv_ln_probs += torch.log(self.inventory_actions[1] - all_states[:, 0] + 1e-8)
            
            old_price_ln_probs = price_dist.log_prob(all_actions[:, 1]).diagonal()
            # old_price_ln_probs += F.logsigmoid(all_actions[:, 1])
            # old_price_ln_probs += F.logsigmoid(all_actions[:, 1])
            # old_price_ln_probs += torch.log(self.price_diff)

        index_list = [i for i in range(len(self.memory))]

        for _ in range(self.epochs):
            # print("Итерация обновления", _)
            batch = random.sample(index_list, self.batch_size)
            states = all_states[batch]
            actions = all_actions[batch]
            local_advantage = advantage[batch]
            
            inv_mu, inv_sigma, prc_mu, prc_sigma = self.c_actor_net(states)
            # print(states[-1])
            # print(inv_sigma)
            inv_dist = torch.distributions.Normal(inv_mu, inv_sigma)
            price_dist = torch.distributions.Normal(prc_mu, prc_sigma)
            inv_ln_probs = inv_dist.log_prob(actions[:, 0]).diagonal()
            # inv_ln_probs += F.logsigmoid(actions[:, 0])
            # inv_ln_probs += F.logsigmoid(-actions[:, 0])
            # inv_ln_probs += torch.log(self.inventory_actions[1] - states[:, 0] + 1e-8)

            price_ln_probs = price_dist.log_prob(actions[:, 1]).diagonal()
            # price_ln_probs += F.logsigmoid(actions[:, 1])
            # price_ln_probs += F.logsigmoid(-actions[:, 1])
            # price_ln_probs += torch.log(self.price_diff)

            # print(inv_ln_probs, old_inv_ln_probs[batch])
            ratio = inv_ln_probs + price_ln_probs - old_inv_ln_probs[batch] - old_price_ln_probs[batch]
            ratio = torch.exp(ratio)
            surr1 = ratio * local_advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * local_advantage
            
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = F.mse_loss(self.c_critic_net(states), td_target[batch].detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # print("#"*40)


class SAC_ActorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
        )

        self.inventory_head_mu = nn.Linear(sloy, 1)
        self.inventory_head_sigma = nn.Linear(sloy, 1)
        self.price_head_mu = nn.Linear(sloy, 1)
        self.price_head_sigma = nn.Linear(sloy, 1)
        

    def forward(self, x):
        y = self.actor(x)
        inv_mu, inv_sigma = self.inventory_head_mu(y), self.inventory_head_sigma(y)
        prc_mu, prc_sigma = self.price_head_mu(y), self.price_head_sigma(y)
        assert not torch.isnan(inv_mu).any(), "inv_mu is NaN"
        assert not torch.isnan(prc_mu).any(), "prc_mu is NaN"
        
        inv_sigma = torch.pow(1 + torch.pow(inv_sigma, 2), 0.5) - 1
        inv_sigma = inv_sigma.clamp(min = 0.01)
        prc_sigma = torch.pow(1 + torch.pow(prc_sigma, 2), 0.5) - 1
        prc_sigma = prc_sigma.clamp(min = 0.01)
        return (inv_mu, inv_sigma, prc_mu, prc_sigma)


class SAC_CriticNet_Q(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, 1)
            )


    def forward(self, x):
        return self.net(x)


class SAC:
    def __init__(
            self,
            state_dim: int,
            inventory_actions: list,
            price_actions: list,
            batch_size: int,
            N_epochs: int,
            MC_samples: int,
            gamma: float,
            actor_lr: float,
            critic_lr: float,
            epochs: int,
            alpha_lr: float,
            target_entropy: float,
            target_scaling: float,
            tau: float,
            cuda_usage: bool,
            dtype = torch.float32,
            ):
        
        self.state_dim = state_dim
        self.inventory_actions = inventory_actions
        self.price_actions = price_actions
        self.cuda_usage = cuda_usage

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dtype = dtype
        self.price_diff = torch.tensor(self.price_actions[1] - self.price_actions[0])

        # Actor and Critic networks
        self.actor = SAC_ActorNet(state_dim).to(self.device)            # Actor
        self.critic_1 = SAC_CriticNet_Q(state_dim, 2).to(self.device)   # Q_1
        self.critic_2 = SAC_CriticNet_Q(state_dim, 2).to(self.device)   # Q_2
        self.target_critic_1 = SAC_CriticNet_Q(state_dim, 2).to(device) # First target Q-network
        self.target_critic_2 = SAC_CriticNet_Q(state_dim, 2).to(device)

        # Q_target parameters are frozen.
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epochs = epochs
        self.memory_size = N_epochs
        self.memory = []
        self.alpha_lr = alpha_lr
        self.log_alpha = torch.tensor(0.0).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = target_entropy
        self.target_scaling = target_scaling

        self.tau = tau
        self.MC_samples = MC_samples
        self.inf_logsigma = torch.log(torch.tensor(1e-4)).to(self.device)
        self.sup_logsigma = torch.log(torch.tensor(1 - 1e-4)).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)


    def __repr__(self):
        return "SAC"


    def _get_state_vector(self, firm_state):
        """Преобразует состояние фирмы в тензор"""
        if not self.cuda_usage:
            inventory = firm_state['current_inventory']
            comp_prices = np.array(firm_state['competitors_prices']).flatten()
            return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)
        else:
            inventory = firm_state['current_inventory']
            comp_prices = np.array(firm_state['competitors_prices']).flatten()
            return torch.tensor([inventory] + comp_prices.tolist(), dtype=self.dtype, device=self.device)


    def suggest_actions(self, firm_state):
        """Возвращает действия (инвентарь, цену)"""
        state = self._get_state_vector(firm_state)
        
        inv_mu, inv_sigma, prc_mu, prc_sigma = self.actor(state)
        
        u_inv = torch.distributions.Normal(inv_mu, inv_sigma).rsample()
        u_prc = torch.distributions.Normal(prc_mu, prc_sigma).rsample()
        inv = state[0] + torch.sigmoid(u_inv/10).clamp(1e-4, 1-1e-4) * (self.inventory_actions[1] - state[0])
        price = self.price_actions[0] + torch.sigmoid(u_prc/10).clamp(1e-4, 1-1e-4) * (self.price_actions[1] - self.price_actions[0])

        return (inv.item(), price.item(), u_inv.item(), u_prc.item())


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
                    next_vec,
                ))
            else:
                self.memory.append((
                    state_vec.cpu(),
                    torch.tensor(actions, dtype = self.dtype).to(self.device),
                    torch.tensor([reward], dtype = self.dtype).to(self.device),
                    next_vec.cpu(),
                ))


    def calc_target_for_target(self, rewards, next_states):
        inv_mu, inv_sigma, prc_mu, prc_sigma = self.actor(next_states)
        inv_dist = torch.distributions.Normal(inv_mu.detach(), inv_sigma.detach())
        price_dist = torch.distributions.Normal(prc_mu.detach(), prc_sigma.detach())
        # u_inv = inv_dist.rsample(sample_shape=(self.MC_samples,)).squeeze().T
        # u_prc = price_dist.rsample(sample_shape=(self.MC_samples,)).squeeze().T
        u_inv = inv_dist.rsample()
        u_prc = price_dist.rsample()

        # inv = states[:, 0].unsqueeze(1) + torch.sigmoid(u_inv/10).clamp(1e-4, 1-1e-4) * (self.inventory_actions[1] - states[:, 0].unsqueeze(1))
        # price = self.price_actions[0] + torch.sigmoid(u_prc/10).clamp(1e-4, 1-1e-4) * self.price_diff

        prom = next_states[:, 0].view(-1, 1)
        inv = prom + torch.sigmoid(u_inv/10).clamp(1e-4, 1-1e-4) * (self.inventory_actions[1] - prom)
        price = self.price_actions[0] + torch.sigmoid(u_prc/10).clamp(1e-4, 1-1e-4) * self.price_diff
        
        # target_input = torch.cat(
        #     [
        #         inv.unsqueeze(2),                                          # (len(self.memory), MC_samples, 1)
        #         price.unsqueeze(2),                                        # (len(self.memory), MC_samples, 1)
        #         states.unsqueeze(1).expand(-1, self.MC_samples, -1)        # (len(self.memory), MC_samples, 2)
        #     ],
        #     dim=2
        # )

        target_input = torch.cat([inv, price, next_states], dim = 1)
        
        Q_tensor = torch.minimum(self.target_critic_1(target_input).detach(),
                                 self.target_critic_2(target_input).detach())
        
        inv_log_prob = inv_dist.log_prob(u_inv)
        inv_log_prob -= F.logsigmoid(u_inv/10).clamp(self.inf_logsigma, self.sup_logsigma)
        inv_log_prob -= F.logsigmoid(-u_inv/10).clamp(self.inf_logsigma, self.sup_logsigma)
        inv_log_prob -= torch.log(self.inventory_actions[1] - prom)
        inv_log_prob += torch.log(torch.tensor(10))

        prc_log_prob = price_dist.log_prob(u_prc)
        prc_log_prob -= F.logsigmoid(u_prc/10).clamp(self.inf_logsigma, self.sup_logsigma)
        prc_log_prob -= F.logsigmoid(-u_prc/10).clamp(self.inf_logsigma, self.sup_logsigma)
        prc_log_prob -= torch.log(self.price_diff)
        prc_log_prob += torch.log(torch.tensor(10))
        
        target_target = Q_tensor - self.log_alpha.exp().detach() * (inv_log_prob + prc_log_prob)
        # target_target = self.target_scaling * E_Q_tensor - (E_inv_log_prob + E_prc_log_prob)
        # target_target = E_Q_tensor - self.alpha.detach() * (E_inv_log_prob + E_prc_log_prob)
        # if t == 10:
        #     print("target_target")
        #     print(target_target)

        return rewards + self.gamma * target_target


    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


    def update(self):
        """Обновление модели на основе данных в памяти"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if not self.cuda_usage:
            all_states = torch.stack([x[0] for x in self.memory])
            all_actions = torch.tensor([x[1] for x in self.memory])
            all_rewards = torch.tensor([x[2] for x in self.memory], dtype = self.dtype).view(-1, 1)
            all_next_states = torch.stack([x[3] for x in self.memory])
        else:
            all_states = torch.stack([x[0] for x in self.memory]).to(self.device)
            all_actions = torch.stack([x[1] for x in self.memory]).to(self.device)
            all_rewards = torch.stack([x[2] for x in self.memory], dim = 1)[0].view(-1, 1).to(self.device)
            all_next_states = torch.stack([x[3] for x in self.memory]).to(self.device)

        index_list = [i for i in range(len(self.memory))]

        for _ in range(self.epochs):
            batch = random.sample(index_list, self.batch_size)
            states = all_states[batch]
            next_states = all_next_states[batch]
            actions = all_actions[batch]
            rewards = all_rewards[batch]

            seen_inv = states[:, 0] + torch.sigmoid(actions[:, 0]/10).clamp(1e-4, 1-1e-4) * (self.inventory_actions[1] - states[:, 0])
            seen_price = self.price_actions[0] + torch.sigmoid(actions[:, 1]/10).clamp(1e-4, 1-1e-4) * self.price_diff
            seen_inv = seen_inv.view(-1, 1)
            seen_price = seen_price.view(-1, 1)

            # Update Q^{target}
            td_target = self.calc_target_for_target(rewards, next_states)
            # if t == 10:
            #     print("self.critic_1(torch.cat([seen_inv, seen_price, states], dim = 1))")
            #     print(self.critic_1(torch.cat([seen_inv, seen_price, states], dim = 1)))
            #     print("self.critic_2(torch.cat([seen_inv, seen_price, states], dim = 1))")
            #     print(self.critic_2(torch.cat([seen_inv, seen_price, states], dim = 1)))
            #     # a += 1
            # target_loss = 0.5 * F.mse_loss(self.target(states), td_target.detach())

            # self.target_optimizer.zero_grad()
            # target_loss.backward()
            # self.target_optimizer.step()

            # # Update both Q-networks
            # td_target = self.calc_target(rewards, next_states)
            critic_1_loss = 0.5 * F.mse_loss(self.critic_1(torch.cat([seen_inv, seen_price, states], dim = 1)), td_target.detach())
            critic_2_loss = 0.5 * F.mse_loss(self.critic_2(torch.cat([seen_inv, seen_price, states], dim = 1)), td_target.detach())

            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            # Update policy network
            inv_mu, inv_sigma, prc_mu, prc_sigma = self.actor(states)
            inv_dist = torch.distributions.Normal(inv_mu, inv_sigma)
            price_dist = torch.distributions.Normal(prc_mu, prc_sigma)
            
            u_inv = inv_dist.rsample(sample_shape=(self.MC_samples,)).squeeze().T
            u_prc = price_dist.rsample(sample_shape=(self.MC_samples,)).squeeze().T

            inv = states[:, 0].unsqueeze(1) + torch.sigmoid(u_inv/10).clamp(1e-4, 1-1e-4) * (self.inventory_actions[1] - states[:, 0].unsqueeze(1))
            price = self.price_actions[0] + torch.sigmoid(u_prc/10).clamp(1e-4, 1-1e-4) * self.price_diff

            target_input = torch.cat(
                [
                    inv.unsqueeze(2),
                    price.unsqueeze(2),
                    states.unsqueeze(1).expand(-1, self.MC_samples, -1)
                ],
                dim=2
            )

            Q_tensor = torch.minimum(self.critic_1(target_input).squeeze(), self.critic_2(target_input).squeeze())
            E_Q_tensor = torch.mean(Q_tensor, axis = 1).view(-1, 1)

            inv_log_prob = inv_dist.log_prob(u_inv.T.unsqueeze(-1)).squeeze(-1).T
            inv_log_prob -= F.logsigmoid(u_inv/10).clamp(self.inf_logsigma, self.sup_logsigma)
            inv_log_prob -= F.logsigmoid(-u_inv/10).clamp(self.inf_logsigma, self.sup_logsigma)
            inv_log_prob -= torch.log(self.inventory_actions[1] - states[:, 0].unsqueeze(1))
            inv_log_prob += torch.log(torch.tensor(10))
            E_inv_log_prob = torch.mean(inv_log_prob, axis = 1).view(-1, 1)

            prc_log_prob = price_dist.log_prob(u_prc.T.unsqueeze(-1)).squeeze(-1).T
            prc_log_prob -= F.logsigmoid(u_prc/10).clamp(self.inf_logsigma, self.sup_logsigma)
            prc_log_prob -= F.logsigmoid(-u_prc/10).clamp(self.inf_logsigma, self.sup_logsigma)
            prc_log_prob -= torch.log(self.price_diff)
            prc_log_prob += torch.log(torch.tensor(10))
            E_prc_log_prob = torch.mean(prc_log_prob, axis = 1).view(-1, 1)

            actor_target = self.log_alpha.exp().detach() * (E_inv_log_prob + E_prc_log_prob) - E_Q_tensor
            # actor_target = (E_inv_log_prob + E_prc_log_prob) - self.target_scaling * E_Q_tensor
            # actor_target = self.alpha.detach() * (E_inv_log_prob + E_prc_log_prob) - E_Q_tensor
            actor_loss = torch.mean(actor_target)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha value
            current_entropy = -(E_inv_log_prob + E_prc_log_prob).mean()
            alpha_loss = self.log_alpha * (current_entropy.detach() - self.target_entropy)
            # alpha_loss = self.alpha * (current_entropy.detach() - self.target_entropy)
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            # Update Q^{target}(\cdot ; \bar{\psi})
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)

