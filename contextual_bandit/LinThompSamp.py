class LinThompSamp:
    """Multi-armed Contextual Bandit Problem -- Thompson.

    Thompson Sampling for Contextual Bandits with Linear Payoffs
    Authors : Shipra Agrawal, Navin Goyal (Microsoft Research India)
    URL : http://jmlr.csail.mit.edu/proceedings/papers/v28/agrawal13.pdf
    We follow the symbols used by
    http://www.cs.cmu.edu/~lizhou/files/contextual_bandit_survey.pdf

    # Input ========================================================================
    param_delta       : parameter 0 < delta <= 1
    param_R           : parameter R >= 0 for R-sub Gaussian
    param_epsilon     : parameter 0 < epsilon < 1
    param_t           : cumulative number of trails
    mtx_context       : contextual vector for each arm
                        (number of arms * dimension of contextual vector)
    mtx_covariance    : the estimated covariance matrix of normal distribution B^(-1)
    vector_mean       : the estimated mean vector of normal distribution
    vector_f          : cumulative selected contextual vector with reward
                        (dimension of contextual vector*1)
    reward            : reward in this trail
    # Output ========================================================================
    param_t           : cumulative number of trails
    mtx_covariance    : the estimated covariance matrix of normal distribution B^(-1)
    vector_mean       : the estimated mean vector of normal distribution
    vector_f          : cumulative selected contextual vector with reward
    """

    def __init__(self, param_delta, param_R, param_epsilon, param_t, mtx_context,...
                 mtx_covariance, vector_mean, vector_f, reward):
        """Init Thompson Sampling for Contextual Bandits."""
        self.param_delta = param_delta
        self.param_R = param_R
        self.param_epsilon = param_epsilon
        self.param_t = param_t
        self.mtx_context = np.array(mtx_context)
        self.mtx_covariance = mtx_covariance
        self.vector_mean = vector_mean
        self.vector_f = vector_f
        self.reward = reward
        self.dim_context = mtx_context.shape[1] # dimension of context vector
        self.param_v2 = (R**2) * 24 * self.dim_context * math.log(1./param_delta)*(1./param.epsilon)
        # self.param_v2 = v^2 the constant of covariance matrix

    def Main(self):
        """Main program of Thompson Sampling for Contextual Bandits."""
        vector_estimean = np.random.multivariate_normal(self.vector_mean, self.param_v2 * self.mtx_covariance,1).T
        expected_reward = np.dot(self.mtx_context, vector_estimean)
        selected_arm = expected_reward.index(max(expected_reward))

        # After receive the reward, Update : mtx_covariance, vector_f, vector_mean, param_t
        # Update mtx_covariance by Sherman Morrison Formula
        temp = np.dot(self.mtx_covariance, mtx_context[selected_arm].T)
        deno = 1 + np.dot(mtx_context[selected_arm], temp)
        self.mtx_covariance = self.mtx_covariance - (np.dot(temp,temp.T) * (1./deno))
        self.vector_f = self.vector_f + (self.reward * mtx_context[selected_arm].T)
        self.vector_mean = np.dot(self.mtx_covariance, self.vector_f)
        param_t = param_t + 1
        return param_t, self.mtx_covariance, self.vector_f, self.vector_mean