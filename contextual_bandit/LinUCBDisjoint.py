class LinUCBDisjoint (object):
""" Multi-armed Contextual Bandit Algorithm -- LinUCB Disjoint version.

LinUCB with Disjoint Linear Models 
ref : A Contextual-Bandit Approach to Personalized News Article Recommendation
authors : Lihong Li, Wei Chu, John Langford (Yahoo! Labs), 
          Robert E.Schapire (Dep. of CS Princeton Univ.)
url: http://arxiv.org/pdf/1003.0146v2.pdf

# Input ========================================================================
param_alpha    : parameter param_alpha >= 0  
param_dim      : dimension of content vector.
param_actions  : number of actions(machines, arms, bandits, articles)
param_T        : total number of trails
mtx_content    : article features (K*self.param_dim, features dim = self.param_dim) (datatype = list)
reward         : reward 
# Output ========================================================================
EXP ? 
"""
    def __init__(self, param_alpha, param_dim, param_actions):
        self.param_alpha = param_alpha
        self.param_dim = param_dim
        self.param_actions = param_actions
        self.Boolean = [False] * param_actions

        strHello = "This is a contextual bandit algorithm -- LinUCB disjoint"
        print strHello

    def train(self, mtx_content):
        """Train phase."""
        for t in range(param_T):
            if t == 0:
                A = []
                b = []
                UCB = np.zeros(self.param_actions, dtype = np.int8)
                for t in range(T):
                    A = A.append(sparse.eye(self.param_dim, dtype=np.int8))
                    b = b.append(np.zeros((self.param_dim, 1), dtype = np.int8))

        [selected_action, selected_content, SHM_h, SHM_deno] = _get_action(self,mtx_content,A,b,UCB)
        reward = get_reward(self, value)
        [A,b] = _update(self, reward, selected_action,selected_content, SHM_h,SHM_deno, A, b):
          

        # return  ========================= (未精簡計算量)
        # output the expected reward of each machine (arm) in this trail\
        EXP = []
        for ii in range(K):
            r_theta = np.dot(A[ii],b)
            EXP.append(np.dot(r_theta,mtx_content[ii]))
        
        return EXP
        




    def _get_action(self, mtx_content, A, b):
        """Compute UCB and Select action with max UCB """
        mtx_content = np.array(mtx_content)
        if t == 0:
        """First trail."""
            for ii in range(K):
                val = np.dot(mtx_content[ii], mtx_content[ii])
                UCB[ii] = (val)

            # Choose arm with max UCB
            selected_action = UCB.index(max(UCB))
            self.Boolean[selected_action] = True
            SHM_h = mtx_content[selected_action]
            SHM_deno = UCB[selected_action]
            
            return selected_action, selected_content, SHM_h, SHM_deno

        else:
        """ More than one trail."""
            UCB = np.zeros(self.param_actions, dtype = np.int8)  
            jj = 0
            for ii in Boolean:
                if ii == True:
                    theta = np.dot(A[jj], b[jj])
                    h = np.dot(mtx_content[jj], A[jj])
                    val = np.dot(h, mtx_content[jj].T)
                    UCB[jj] = np.dot(theta,mtx_content[jj]) + param_alpha * math.sqrt(val))
                    jj = jj + 1
                else:
                    val = np.dot(mtx_content[jj], mtx_content[jj])
                    UCB[jj] = val                    
                    jj = jj + 1

            selected_action = UCB.index(max(UCB))
            if Boolean[selected_action] == True:
                SHM_h = np.dot(mtx_content[selected_action], A[selected_action])
                SHM_deno = 1 + np.dot(SHM_h, mtx_content[selected_action].T)
            else:
                SHM_h = mtx_content[selected_action]
                SHM_deno = 1 + np.dot(mtx_content[selected_action],mtx_content[selected_action])
            
            selected_content = mtx_content[selected_action]
            self.Boolean[selected_action] = True
            
            return selected_action, selected_content, SHM_h, SHM_deno


    def get_reward(self, value):
            reward = value
            return reward

    def _update(self, reward, selected action,selected_content, SHM_h,SHM_deno, A, b):
        """ Update matrix A and vector b."""
        ind = selected_actions
        A[ind] = A[ind] - (1./SHM_deno)*np.dot(SHM_h.T , SHM_h)
        b[ind] = b[ind] + reward * selected_content.T 
        return A, b
