Class EXP4(object):
    """Multi-armed Contextual Bandit Algorithm -- EXP4.

    EXP4 : Exponential Weighted Algorithm for Exploration and Exploitation 
           using Expert Advices
    Upper bound for regret(T) = O(T+K*lnN)

    # Input ========================================================================
    param_alpha     : parameter alpha>0
    param_gamma     : parameter 1 >= gamma >= 0
    param_T         : total number of trails
    param_experts   : number of experts
    param_actions   : number of actions (machines,arms,bandits,articles)
    mtx_experts     : expert advices matrix
                      (dim of advices(k)*number of experts(N))
    reward          : reward in this trail
    # Output ========================================================================
    selected machine in t th trail??
    """

    def __init__(self,param_alpha,param_gamma,param_experts,param_actions):
        """Init EXP4 for Contextual Bandit."""
        self.param_alpha = param_alpha
        self.param_gamma = param_gamma
        self.param_actions = param_actions
        self.weight_of_advices = np.ones(param_experts, dtype=np.int)
        self.weighted_reward = np.zeros(param_experts, dtype=np.int)
        strHello = "This is a contextual bandit algorithm -- EXP4"
        print strHello

    def train(self, param_T, mtx_experts, value):
        """Train phase."""
        for t in range(param_T): 
            [selected_action, prob_selected_action] = _get_action(self,mtx_experts)
            reward = get_reward(self,value)
            [self]= _update(self,selected_action, prob_selected_action, reward, mtx_experts)

        def _get_action(self,mtx_experts):
            temp = np.dot(mtx_experts,self.weight_of_advices)*(1./sum(self.weight_of_advices))
            prob_each_action = (1-self.param_gamma)*temp + self.param_gamma./self.param_actions
            culmulative_prob = np.zeros(self.param_actions, dtype=np.int)
            val = rand.random()
            culmulative_prob[0] = prob_each_action[0]
            for ii in range(1,K):
                culmulative_prob[ii] = culmulative_prob[ii-1] + prob_each_action[ii]
            for jj in range(K):
                if val < culmulative_prob[jj]
                    selected_action = jj
                    break
            prob_selected_action = prob_each_action[selected_action]
            return selected_action, prob_selected_action

        # @get_reward.setter
        def get_reward(self, value):
            reward = value
            return reward

        # @abstractmethod
        def _update(self, selected_action, prob_selected_action, reward, mtx_experts):
            """Update weight of experts' advices."""
            ind = selected_action
            param = self.param_gamma./param_actions
            self.weighted_reward += (reward./prob_for_each_arm[ind]) * mtx_expert[ind]
            self.weight_of_advices = np.exp(param * self.weighted_reward)
            return self.weighted_reward, self.weight_of_advices