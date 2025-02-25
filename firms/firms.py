### для venv: & ".\.venv\Scripts\Activate"
### для папки: cd DRL_pricing
### для докера: pip freeze > requirements.txt

### Здесь будут нужные функции для использования в промежуточных частях кода

# Imported relevant python libraries
import numpy as np
import matplotlib.pyplot as plt
import random

class epsilon_greedy:
     
    def __init__(
            self,
            eps: float,
            Q_list: list,
            action_list: list,            
			):

        assert len(Q_list) == len(action_list), "Length doesn't match!"

        self.eps = eps
        self.Q_list = Q_list
        self.action_list = action_list
            
    def hello(self):
        print("OK!", self.eps)
    

# ex = epsilon_greedy(0.5, [], [])

# # Total number of bandit problems
# banditProblems = 20
# # Total number of arms in each bandit problem
# k=2
# # Total number of times to pull each arm
# armPulls=200

# # True means generated for each arms for all the bandits
# trueMeans=np.random.normal(0, 1, (banditProblems, k))
# # Storing the true optimal arms in each bandit
# trueOptimal=np.argmax(trueMeans, 1)
# # Each row represents a bandit problem

# # Array of values for epsilon
# epsilon = [0, 0.1]
# col = ['r', 'g']

# # Adding subplots to plot and compare both plots simultaneously
# plotFirst=plt.figure().add_subplot(111)
# plotSecond=plt.figure().add_subplot(111)

# for x in range(len(epsilon)) :

# 	print('The present epsilon value is : ', x)

# 	# Storing the predicted reward
# 	Q = np.zeros((banditProblems,k))
# 	# Total number of times each arms is pulled
# 	N = np.ones((banditProblems,k))
# 	# Assigning initial random arm pulls
# 	initialArm = np.random.normal(trueMeans, 1)

# 	rewardEps=[]
# 	rewardEps.append(0)
# 	rewardEps.append(np.mean(initialArm))
# 	rewardEpsOptimal = []

# 	for y in range(2, armPulls+1) :
# 		# All rewards in this pull/time-step
# 		rewardPull=[] 
# 		# Number of pulss of best arm in this time step
# 		optimalPull = 0 
# 		for z in range(banditProblems) :

# 			if random.random() < epsilon[x] :
# 				i=np.random.randint(k)
# 			else:
# 				i=np.argmax(Q[z])
			
# 			# To calculate % optimal action
# 			if i == trueOptimal[z]: 
# 				optimalPull = optimalPull + 1

# 			rewardTemp = np.random.normal(trueMeans[z][i], 1)
# 			rewardPull.append(rewardTemp)
# 			N[z][i] = N[z][i] + 1
# 			Q[z][i] = Q[z][i] + (rewardTemp - Q[z][i])/N[z][i]

# 		rewardAvgPull = np.mean(rewardPull)
# 		rewardEps.append(rewardAvgPull)
# 		rewardEpsOptimal.append(float(optimalPull)*100/banditProblems)
# 	plotFirst.plot(range(0, armPulls + 1), rewardEps, col[x])
# 	plotSecond.plot(range(2, armPulls + 1), rewardEpsOptimal, col[x])

# #plt.ylim(0.5,1.5)
# plotFirst.title.set_text('epsilon-greedy : Average Reward Vs Steps for 10 arms')
# plotFirst.set_ylabel('Average Reward')
# plotFirst.set_xlabel('Steps')
# plotFirst.legend(("\epsilon="+str(epsilon[0]),"\epsilon="+str(epsilon[1])),loc='best')
# plotSecond.title.set_text('\epsilon-greedy : \% Optimal Action Vs Steps for 10 arms')
# plotSecond.set_ylabel('\% Optimal Action')
# plotSecond.set_xlabel('Steps')
# plotSecond.set_ylim(0,100)
# plotSecond.legend(("\epsilon="+str(epsilon[0]),r"\epsilon="+str(epsilon[1])),loc='best')
# plt.show()