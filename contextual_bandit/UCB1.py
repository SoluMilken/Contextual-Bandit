''' 
Multi-armed Bandit Problem
Upper Confidence Bound 1 (UCB1)
# Input ========================================================================
T         : number of trails   
K         : number of machines(arms,bandits,articles)
r         : reward 
# Output ========================================================================
selected machine in t th trail
'''
# initialization
from scipy import sparse
import math
counter = np.ones(K, dtype=np.int)
icounter = np.ones(K, dtype=np.int)
#accumulated_reward = np.zeros(K,dtype=np.int)

# Play each mathine once
accumulated_reward = r_vector # the reward of each machine @@

for t in range(T):
    P = (accumulated_reward*np.array([1./(t+1)]*K)) + math.sqrt(np.array([2*math.log(t+1)]*K)*icounter)
    ind = P.index(max(P))
    counter[ind] = counter[ind] + 1
    icounter[ind] = 1./counter[ind]
    
    print ind