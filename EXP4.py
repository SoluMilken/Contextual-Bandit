class EXP4(object):
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

    def __init__(self, param_alpha, param_gamma, param_experts, param_actions):
        """Init EXP4 for Contextual Bandit.
        """
        self.param_alpha = param_alpha
        self.param_gamma = param_gamma
        self.param_actions = param_actions
        self.weight_of_advices = np.ones(param_experts, dtype=np.int)
        self.weighted_reward = np.zeros(param_experts, dtype=np.int)

    def start_train(self, param_T, mtx_experts):
        """Train phase.
        """
        self.count = 0
        self.param_T = param_T
        self.mtx_experts = mtx_experts
        selected_action, prob_selected_action = self._get_action(self.mtx_experts)
        return selected_action, prob_selected_action

    def iterate_reward(self, reward):
        """Please Run start_train before iterate_reward THX! 
        """
        if self.count <= self.param_T:
            self._update(selected_action, prob_selected_action, reward, self.mtx_experts)
            selected_action, prob_selected_action = self._get_action(self.mtx_experts)
            return selected_action, prob_selected_action
        else:
            return None, None

    def _get_action(self, mtx_experts):
        """
        Select an action based on the certain probability
        Args:
            mtx_experts (matrix) : experts advice matrix 
        Returns:
            selected_action        : the chosen action
            prob_selected_action   : the probability of the chosen action
        Examples:
            exp4._get_action(...)
        """
        # Maybe you should describe more on this part
        temp = np.dot(mtx_experts, self.weight_of_advices) * (1. / sum(self.weight_of_advices))
        prob_each_action = (1 - self.param_gamma) * temp + self.param_gamma. / self.param_actions
        culmulative_prob = np.zeros(self.param_actions, dtype=np.int)
        val = rand.random()
        culmulative_prob[0] = prob_each_action[0]
        # Or more comments on this block
        for i in range(1, K): # K from where?
            culmulative_prob[i] = culmulative_prob[i-1] + prob_each_action[i]
        for j in range(K):
            if val < culmulative_prob[j]
                selected_action = j
                break
        prob_selected_action = prob_each_action[selected_action]
        return selected_action, prob_selected_action

    # @abstractmethod
    def _update(self, selected_action, prob_selected_action, reward, mtx_experts):
        """Update weight of experts' advices.
        Args:
            param1 (int): The first parameter.
            param2 (str): The first parameter.
        Returns:
            bool: True if successful, False otherwise.
        Examples:
            exp4._get_action(...)
        """
        ind = selected_action
        param = float(self.param_gamma) / param_actions
        self.weighted_reward += (float(reward) / prob_for_each_arm[ind]) * mtx_expert[ind]
        self.weight_of_advices = np.exp(param * self.weighted_reward)