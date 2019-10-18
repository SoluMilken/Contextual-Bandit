''' 
Multi-armed Bandit Problem
Epoch-Greedy 
# Input ========================================================================
c         : parameter c>0
d         : parameter d belongs to (0,1)
T         : number of trails   
K         : number of machines(arms,bandits,articles)
r         : reward 
# Output ========================================================================
selected machine in t th trail
'''
import random as rand
import numpy as np

def Epoch_Greedy(c,d,T,K,r)
    accumulated_reward = np.zeros(K, dtype=np.int)

    for t in range(T):
        if t == 0:
            # first trail
            ind = rand.randrange(K)
            accumulated_reward[ind] = r  # r: reward in first trail
        else:
            # more than one trail
            epoch = min(1,(c*K)/(d*d*(t+1)))
            val = rand.random()
            if val < epoch:
                ind = rand.randrange(K)
                accumulated_reward[ind] = accumulated_reward[ind] + r # r: reward for ind machine in trail t
            else:
                ind = accumulated_reward.index(max(accumulated_reward))
                accumulated_reward[ind] = accumulated_reward[ind] + r # r: reward for ind machine in trail t
            