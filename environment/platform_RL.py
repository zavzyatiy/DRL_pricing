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
import random
from copy import deepcopy

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class no_platform:
    def __init__(smthing):
        pass


    def __repr__(self):
        return "None"


    def cache_data(self, state):
        pass
    

    def suggest(self, p):
        return (0, 0, 0, 1)


    def update(self):
        pass


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
        
        self.weight = weight
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

        demand = state["doli"] * state["boosting"]
        self.memory.append(demand)
    

    def suggest(self, plat_info):
        p = plat_info["p"]
        first = (self.p - p)/self.diff
        if len(self.memory) > 0:
            second = np.mean(np.array(self.memory).T, axis = 1) / self.C
        else:
            second = np.ones(self.n)
        res = self.weight * first + (1 - self.weight) * second
        e_res = np.exp(res)
        return (0, 0, self.weight, self.n * e_res/np.sum(e_res))
        # w = self.weight
        # res = w * torch.tensor(first.tolist()) + (1 - w) * torch.tensor(second.tolist())
        # e_res = res.exp()
        # return self.n * e_res/(e_res).sum()


    def update(self):
        pass


def compute_advantage(gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)


class PPO_C_ActorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.c_actor_net = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
            nn.Linear(sloy, sloy),
            nn.LayerNorm(sloy),
            nn.ReLU(),
        )

        self.alpha_head_mu = nn.Linear(sloy, 1)
        self.alpha_head_sigma = nn.Linear(sloy, 1)
        

    def forward(self, x):
        y = self.c_actor_net(x)
        alpha_mu, alpha_sigma = self.alpha_head_mu(y), self.alpha_head_sigma(y)
        # assert not torch.isnan(alpha_mu).any(), "alpha_mu is NaN"
        if torch.isnan(alpha_mu).any():
            print(x)
            print(y)
            print(alpha_mu, alpha_sigma)
            print("alpha_mu is NaN")
        # alpha_sigma = torch.pow(1 + torch.pow(alpha_sigma, 2), 0.5) - 1
        alpha_sigma = torch.sqrt(1 + torch.pow(alpha_sigma, 2)) - 1
        alpha_sigma = alpha_sigma.clamp(min = 0.01)
        return (alpha_mu, alpha_sigma)


class PPO_C_CriticNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        sloy = 256 # 256
        
        self.c_critic_net = nn.Sequential(
            # nn.LayerNorm(input_dim),
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


class dynamic_weights:
    def __init__(
            self, 
            state_dim: int,
            d_memory_size: int,
            n: int,
            p_inf: float,
            p_max: float,
            C: float,
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
        self.d_memory_size = d_memory_size
        self.p = p_max
        self.diff = p_max - p_inf
        self.n = n
        self.C = C
        
        self.d_memory = []
        self.index_list = [i for i in range(N_epochs)]
        self.cuda_usage = cuda_usage

        if cuda_usage:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.dtype = dtype

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
        return "dynamic_weights"


    def _get_state_vector(self, plat_state):
        
        first = plat_state['first'][:-1]
        second = plat_state['second'][:-1]
        # stock = plat_state['stock'] / self.C
        # inv = plat_state['inv'] / self.C
        return torch.tensor(first.tolist() + second.tolist(), dtype=self.dtype, device=self.device) # inv.tolist() + stock.tolist() + 


    def suggest(self, plat_state):
        p = plat_state["p"]
        first = (self.p - p)/self.diff
        if len(self.d_memory) > 0:
            second = np.mean(np.array(self.d_memory).T, axis = 1) / self.C
        else:
            second = np.ones(self.n)

        state = self._get_state_vector({"first": first, "second": second,
                                        "stock": plat_state["stock"], "inv": plat_state["inv"]})
        
        alpha_mu, alpha_sigma = self.c_actor_net(state)
        
        u_alpha = torch.distributions.Normal(alpha_mu, alpha_sigma).sample()
        weight = torch.sigmoid(u_alpha/10).clamp(1e-4, 1-1e-4).item()
        u_alpha = u_alpha.item()

        res = weight * first + (1 - weight) * second
        e_res = np.exp(res)
        boosting = self.n * e_res/np.sum(e_res)

        return (first, second, u_alpha, boosting)


    def cache_data(self, state):
        self.cache_experience(state)

        if len(self.d_memory) >= self.d_memory_size:
            self.d_memory.pop(0)

        demand = state["doli"] * state["boosting"]
        self.d_memory.append(demand)


    def cache_experience(self, state):
        """Сохраняет опыт в replay buffer"""
        state_vec = self._get_state_vector(state)
        actions = state["action"]
        reward = state["plat_pi"]
        
        if len(state_vec) == self.state_dim:
            if not self.cuda_usage:
                self.memory.append((
                    state_vec,
                    actions,
                    reward,
                ))
            else:
                self.memory.append((
                    state_vec.cpu(),
                    torch.tensor([actions], dtype = self.dtype).to(self.device),
                    torch.tensor([reward], dtype = self.dtype).to(self.device),
                ))


    def update(self):
        """Обновление модели на основе данных в памяти"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if not self.cuda_usage:
            all_states = torch.stack([x[0] for x in self.memory])
            all_actions = torch.tensor([x[1] for x in self.memory])[1:]
            all_rewards = torch.tensor([x[2] for x in self.memory], dtype = self.dtype).view(-1, 1)[1:]
            # all_next_states = all_states[1:]
            # all_states = all_states[:-1]
        else:
            all_states = torch.stack([x[0] for x in self.memory]).to(self.device)
            all_actions = torch.stack([x[1] for x in self.memory]).to(self.device)[1:]
            all_rewards = torch.stack([x[2] for x in self.memory], dim = 1)[0].view(-1, 1).to(self.device)[1:]
            # all_next_states = all_states[1:]
            # all_states = all_states[:-1]
        
        # print(all_next_states[:5])
        # print(self.c_critic_net(all_next_states[:5]))
        # print(all_rewards[:5])
        prom = self.c_critic_net(all_states)
        td_target = all_rewards + self.gamma * prom[1:]
        td_delta = td_target - prom[:-1]
        # all_next_states = all_states[1:]
        all_states = all_states[:-1]
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # print(advantage[:5])
        # a += 1

        with torch.no_grad():
            alpha_mu, alpha_sigma = self.c_actor_net(all_states)
            alpha_dist = torch.distributions.Normal(alpha_mu.detach(), alpha_sigma.detach())
            old_alpha_ln_probs = alpha_dist.log_prob(all_actions).diagonal()

        for _ in range(self.epochs):
            batch = random.sample(self.index_list, self.batch_size)
            states = all_states[batch]
            actions = all_actions[batch]
            local_advantage = advantage[batch]
            
            alpha_mu, alpha_sigma = self.c_actor_net(states)

            alpha_dist = torch.distributions.Normal(alpha_mu, alpha_sigma)
            alpha_ln_probs = alpha_dist.log_prob(actions).diagonal()
            
            ratio = alpha_ln_probs - old_alpha_ln_probs[batch]
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
        
        self.memory = [self.memory[-1]]

