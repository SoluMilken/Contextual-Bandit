''' 
LinUCB with Disjoint Linear Models 
ref : A Contextual-Bandit Approach to Personalized News Article Recommendation
authors : Lihong Li, Wei Chu, John Langford (Yahoo! Labs), 
          Robert E.Schapire (Dep. of CS Princeton Univ.)
url: http://arxiv.org/pdf/1003.0146v2.pdf
# Input ========================================================================
alpha >=0 : parameter   
t         : t th trails
K         : number of machines(arms,bandits,articles)
X         : article features (K*d, features dim = d) (datatype = list)
r         : reward 
# Output ========================================================================
EXP ? 
'''
# initialization
import numpy as np
from scipy import sparse
import math
d = len(X[0])  # dimension of features
K = len(X)     # number of machines(arms,bandits,articles)
A = []
b = []
Boolean = [False]*K

# Training Phase
for t in range(T):
    A = A.append(sparse.eye(d, dtype=np.int8))
    b = b.append(sparse.csr_matrix((d,1),dtype = np.int8))
    
    [P,H,deno] = inner_LinUCB(alpha,t,d,K,X,A,b,Boolean)
    
    # Choose arm with max UCB
    ind = P.index(max(P))
    Boolean[ind] = True
    
    # Update matrix A and vector b 
    A[ind] = A[ind] - (np.dot(H[ind].T,H[ind]))/(1+deno[ind])
    b[ind] = b[ind] + r*X[ind].T  #(reward "r"要從外面抓進來)
    
    # return (1)the selected machine in this trail
    print ind+1
    
    # return  ========================= (未精簡計算量)
    # (2) the expected reward of each machine (arm) in this trail\
    EXP = []
    for ii in range(K):
        r_theta = np.dot(A[ii],b)
        EXP.append(np.dot(r_theta,X[ii]))
    
    print EXP
        
    
    
    
    
# Compute UCB for each machine (arm) ===============================
def inner_LinUCB(alpha,t,d,K,X,A,b,Boolean)
    # inner training phase
    X = np.array(X)
    if t == 0
    # first trail
        P = []
        for ii in range(K):
            val = np.dot(X[ii],X[ii])
            P.append(val) 
            H.append(X[ii])
            deno.append(val)
            
    else
    # More than one trail
        P = []  # UCB
        H = []
        deno = []
        jj = 0
        for ii in Boolean:
            if ii==True:        
                theta = np.dot(A[jj],b[jj])
                h = np.dot(X[jj],A[jj])
                val = np.dot(h,X[jj].T)
                P.append(np.dot(theta,X[jj])+ alpha*math.sqrt(val))
                H.append(h)
                deno.append(val)
                jj = jj + 1
            else:
                val = np.dot(X[jj],X[jj])
                P.append(val) 
                H.append(X[jj])
                deno.append(val)
                jj = jj + 1
    return P,H,deno
        