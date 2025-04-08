### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### удалить локальные изменения: git reset --hard HEAD
### для докера: pip freeze > requiremnts.txt

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


class fixed_weights:
    def __init__(
            self,
            weights: list,
            memory_size: int,
            n: int,
            p_inf: float,
            p_max: float,
            C: float,
			):
        
        weights = np.array(weights)
        assert np.sum(weights) == 1

        self.weights = weights
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

        self.memory.append(state)
    

    def suggest(self, p):
        first = (self.p - p)/self.diff
        if len(self.memory) > 0:
            second = np.mean(np.array(self.memory).T, axis = 1) / self.C
        else:
            second = np.ones(self.n)
        res = self.weights[0] * first + self.weights[1] * second
        e_res = np.exp(res)
        return self.n * e_res/np.sum(e_res)


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
        assert not torch.isnan(alpha_mu).any(), "alpha_mu is NaN"
        alpha_sigma = torch.pow(1 + torch.pow(alpha_sigma, 2), 0.5) - 1
        alpha_sigma = alpha_sigma.clamp(min = 0.01)
        return (alpha_mu, alpha_sigma)


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


class PPO_C_Platform:
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
        return "PPO_C_Platform"


    def _get_state_vector(self, plat_state):
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

