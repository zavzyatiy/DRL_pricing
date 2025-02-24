### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

### Здесь будут нужные функции для использования в промежуточных частях кода

import numpy as np

a = [1, 5, 4, 3, 2]
b = [np.exp(x) for x in a]
S = sum(b)
c = [x/S for x in b]
print(c)