''' 
Contextual Bandit Problem
EXP4 : Exponential Weighted Algorithm for Exploration and Exploitation 
       using Expert Advices
Upper bound for regret(T) = O(T+K*lnN)
# Input ========================================================================
alpha     : parameter alpha>0
T         : number of trails   
K         : number of machines(arms,bandits,articles)
E         : expert advices matrix (dim of advices(k)*number of experts(N))
r         : reward 
# Output ========================================================================
selected machine in t th trail
'''
strHello = "This is a contextual bandit algorithm -- EXP4"
print strHello

def EXP4(gamma,T,K,E,r)
    #initialization
    N = len(E[0])       # number of experts
    weight_of_advices = np.ones(N, dtype=np.int)
    weighted_reward = np.zeros(N, dtype=np.int)
    
    for t in range(T): 
        [weighted_reward,weight_of_advices] = inner_EXP4(gamma,K,E,weighted_reward,r)
        # Output
        

def inner_EXP4(gamma,K,E,weighted_reward,weight_of_advices,r)
    prob_for_each_arm = ((1-gamma)./sum(weight_of_advices))*np.dot(E,weight_of_advices) + gamma./K

    accumulated_prob = np.zeros(K, dtype=np.int)
    val = rand.random()
    accumulated_prob[0] = prob_for_each_arm[0]
    for ii in range(1,K):
        accumulated_prob[ii] = accumulated_prob[ii-1] + prob_for_each_arm[ii]
    for jj in range(K):
        if val < accumulated_prob[jj]
            ind = jj
            break
    weighted_reward = weighted_reward + (r./prob_for_each_arm[ind])*E[ind]  # r: reward in trail t
    weight_of_advices = np.exp((gamma./K)*weighted_reward) 
    
    return weighted_reward,weight_of_advices                                                 