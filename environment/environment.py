### начальные условия: инициализация весов для всех алгоритмов
### для env: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing

import random as rnd
import numpy as np
import torch

### иницализация random.seed

rnd.seed(42)
torch.manual_seed(42)

### инициализация гиперпараметров: n, m, \delta, \gamma,
### c_i, h^+, v^-, \eta

print(rnd.randint(1, 10))
