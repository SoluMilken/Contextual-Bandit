"""
Multi-armed Bandit Problem
Exp3 : Exponential Weighted Algorithm
# Input ========================================================================
gamma     : parameter 0<a<=1
T         : number of trails   
K         : number of machines(arms,bandits,articles)
r         : reward 
# Output ========================================================================
selected machine in t th trail
"""
import numpy as np
import math

def EXP3(gamma,T,K,r)
    accumulated_reward = np.ones(K, dtype=np.int)
    for t in range(T): 
        if t == 0:
            # first trail
            ind = rand.randrange(K)
            accumulated_reward[ind] = math.exp((gamma./K)*r)  # r: reward in first trail
            deno = K-1 + accumulated_reward[ind]
        else:
            # more than one trail
            prob_for_each_arm = ((1-gamma)/deno)*accumulated_reward + gamma./K
            
            accumulated_prob = np.zeros(K, dtype=np.int)
            val = rand.random()
            accumulated_prob[0] = prob_for_each_arm[0]
            for ii in range(1,K):
                accumulated_prob[ii] = accumulated_prob[ii-1] + prob_for_each_arm[ii]
            for jj in range(K):
                if val < accumulated_prob[jj]
                    ind = jj
                    break
            deno = deno - accumulated_reward[ind]
            accumulated_reward[ind] = accumulated_reward[ind]*math.exp((gamma*r)./(K*prob_for_each_arm[ind])    # r: reward in trail t
            deno = deno + accumulated_reward[ind]
    
    return accumulated_reward                                                   