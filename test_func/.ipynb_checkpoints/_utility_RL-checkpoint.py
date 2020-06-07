#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
##########################################################################
from ._utility import *
from ._uti_basic import *
##########################################################################
param_grid = {'max_depth': [2, 6, 10], 'min_samples_leaf': [5, 10, 20]}
n_jobs = multiprocessing.cpu_count() 
##########################################################################
def change_rate(y_old, y_new):
    return norm(y_old - y_new)**2 / norm(y_old)**2

def flatten(l): 
    # list of sublist -> list
    return [item for sublist in l for item in sublist]

def cum_r(rewards, gamma):
    """ rewards -> culmulative reward
    """
    return sum(
        map(operator.mul, [gamma ** j for j in range(len(rewards))], rewards))
cum_rewards = cum_r
##########################################################################

#%% Prepare training data for the Fitted-Q: based on state (mul-J) transition and observed rewards

def ObsAct2State(obs, actions, t, J, multiple_N = False):
    """ Based on our discussion on 12/03, to form a lag-J states from history obs and A
    For RL purpose. The testing part is clear.
    To make A_t, we need to define S_t, which is (with lag-J) (e.g., when lag-1, S_t+1 only depneds on X_t and A_t): 
    O_(t-J + 1), A_(t - J+1), ..., O_t
    """
    if not multiple_N:
        if J == 1:
            s = obs[t, :].ravel(order='C')
        else:
            s = np.hstack([obs[(t - J + 1): t, :], actions[(t - J + 1):t]]).ravel(order='C')
            s = np.append(s, obs[t, :].ravel())
        return s
    else: # obs: 3 * T * N
        N = obs.shape[2]
        dim_obs = 3
        if J == 1:
            s = obs[:, t, :]
        else: # target: (4 * J_Q - 1) * N
            s = np.vstack(([
                obs[:, (t - J + 1 ):t, :],
                actions[(t - J + 1):t, :].reshape((1, J - 1, N))])) # extend_dim for first one
            s = s.reshape(((dim_obs + 1) * (J - 1), N), order = 'F')
            obs_0 = obs[:, t, :] # 3 * N
            s = np.vstack([s, obs_0])
        return s # dim * N
            


def MDP2Trans(MDPs, J, action_in_states = False, combined = True):
    """
    Input: a list (len-N) of trajectory [state matrix [T * 3], actions, rewards] - I need to modify evaluate.ipynb
    Output: a list of (s,a,s',u) (combined together)
    """
    def MDP2Trans_one_traj(i):
        obs, actions, utilities = MDPs[i]
        T = obs.shape[0]
        result = []
        for t in range(J - 1, T - 1):
            s = ObsAct2State(obs, actions, t, J)
            ss = ObsAct2State(obs, actions, t + 1, J)
            
            a = actions[t]
            u = utilities[t]
            result.append([s, a, ss, u])
        return result
    r = rep_seeds(MDP2Trans_one_traj, len(MDPs) - 1)
    if combined:
        return flatten(r) # put every patient together; not into a metrix
    else:
        return r

##########################################################################
""" Fitted Q
1. fit (x,a) -> q(x,a)
2. update q(a,x) = max_{a'}(q(a',x')) + gamma * r  # (x',r) is observed
"""
##########################################################################
# %% Main functions for Fitted-Q
def NFQ(PatternSets, gamma, RF_paras = [3,20], n_trees = 200, threshold = 1e-5, initialize = "mine"):
    """ Learn optimal Q function from batch data (RF + fitted-Q)
    Input: a list of (s,a,s',u)
    Output: Q function
    """
    rseed(0); npseed(0)
    ### Preparing training data
    s, a, ss, r = [np.array([a[i] for a in PatternSets]) for i in range(4)]
    a = np.array([a[1] for a in PatternSets]).reshape((-1, 1))
    range_a = np.unique(a)
    x_train = np.hstack((s, a))
    
    ### Initialization
    init_y = r * (1 / (1 - gamma)) # based on the series result
    is_CV = False
    if RF_paras == "CV": 
        rseed(0); npseed(0)
        is_CV = True
        rfqr = RF(random_state = 0, n_estimators = n_trees)
        gd = GridSearchCV(estimator = rfqr, param_grid = param_grid, cv = 3, n_jobs = n_jobs, verbose=0)
        gd.fit(x_train, init_y.ravel())
        RF_paras = gd.best_params_
        RF_paras = [RF_paras['max_depth'], RF_paras['min_samples_leaf']]
    
    rseed(0); npseed(0)
    max_depth, min_samples_leaf = RF_paras
    Q = RF(max_depth = max_depth, random_state = 0, n_estimators = n_trees, min_samples_leaf =   
           min_samples_leaf, n_jobs = n_jobs, 
           verbose = 0) 
    Q.fit(x_train, init_y.ravel())
    
    ### Iterations
    y_old = init_y.copy()
    # update the estimated Q
    rep, epsilon = 0, 100
    while(epsilon > threshold and rep < 100): # 200 before
        rseed(0); npseed(0)
        y_train = UpdatedValues(ss, range_a, r, Q, gamma)
        epsilon = change_rate( y_old = y_old, y_new = y_train)
        Q.fit(x_train, y_train.ravel()) 
        y_old = y_train.copy()
        rep += 1
    return Q


def UpdatedValues(ss, range_a, r, Q, gamma):
    """ Update the estimated optimal v(s,a) with the fitted Q function
    Input: 
        PatternSets = a list of (s,a,s',r), Q
        ss0, ss1: (s', 0), (s', 1) --- just for lasy
        r: observed rewards
        Q: for values at next states
    Output: ((s,a),v), where v = r + gamma * max_a' Q(s',a'); 0/1 action in this example.
    """
    v_as = []
    N = ss.shape[0]
    for a in range_a:
        ss_a = np.hstack((ss, np.ones((N, 1)) * a ))
        v_a = Q.predict(ss_a)
        v_as.append(v_a.reshape(N, 1))
    v_max = np.amax(np.hstack(v_as), 1)
    Q_new = r.reshape(N, 1) + gamma * v_max.reshape(N, 1)
    return Q_new


def Estpolicy(Q_func, range_a):
    """ Q function to Policy
    Input:
        Q-function and the range of available actions
    Output:
        The optimal action policy  (discrete) at this state [given a state, output an action]
    """
    def policy(s, debug = 0): 
        """
        Input: s [N * dx]
        Output: actions [N * 1]
        """   
        rseed(0); npseed(0)
        N  = s.shape[0]
        v_as = []
        for a in range_a:
            s_a = np.hstack([s,np.repeat(a, N).reshape(-1,1)])
            v_a = Q_func.predict(s_a)
            v_as.append(v_a.reshape(-1, 1))
        v_as = np.round(np.hstack(v_as), 4)
        actions = np.array([range_a[i] for i in np.argmax(v_as, 1)]).reshape(-1, 1)
        if debug == 1:
            print(v_as - v_as[:,1].reshape(-1,1), DASH, actions)
        return actions

    return policy

##########################################################################
##########################################################################
def UpdatedValues_eval(ss, policy, J, r, Q, gamma):
    """ Version of 1-step forward in Evaluations
    """
    dx = ss.shape[1]
    sss = ss[:,(dx - (4 * J - 1)):dx]
    As = policy(sss)
    sa = np.hstack([ss,As])
    return gamma * Q.predict(sa).reshape(-1,1) + r.reshape(-1,1)

def FQE(PatternSets, Q_func, J, gamma = 0.9, RF_paras = [3, 20], n_trees = 200, 
                         threshold = 1e-4):
    """ 
    Fitted-Q Evaluation for off-policy evaluation (OPE) in REAL DATA
        
        1. fit RF q: (x,a) -> value
        2. update the value function of policy:
            q_policy(x, a) = gamma * q(x', policy(x'[, (dx - J): dx])) + r
            
    3. q_policy(x, x[, (dx - J): dx])
    
    Input: 
        PatternSets: a list of (s, a, s', u) [have been transformed]
        
    Output: V function

    """
    rseed(0); npseed(0)
    
    # Preparing training data
    s_bef, a_bef, ss_bef, r_bef = [np.array([a[i] for a in PatternSets]) for i in range(4)]
    a_bef = a_bef.reshape(-1, 1)
    range_a = np.unique(a_bef)
    
    policy = Estpolicy(Q_func, range_a)
    time = now()
    
    dx = s_bef.shape[1]    
    s1 = s_bef[:,(dx - (4 * J - 1)):dx].copy()
    As = policy(s1)
    selected = (As == a_bef)

    s2, a2, ss2, r2 = [], [], [], []
    for i in range(s_bef.shape[0]):
        if selected[i, 0]:
            s2.append(s_bef[i,])
            a2.append(a_bef[i,])
            ss2.append(ss_bef[i,])
            r2.append(r_bef[i,])
    s, a, ss, r = np.vstack(s2).copy(), np.vstack(a2).copy(), np.vstack(ss2).copy(), np.vstack(r2).copy()
    
    
    ### Initialization
    x_train = np.hstack((s, a))
    init_y = r * (1 / (1 - gamma))
    if RF_paras == "CV":
        rseed(0); npseed(0)
        rfqr = RF(random_state = 0, n_estimators = n_trees)
        gd = GridSearchCV(estimator=rfqr, param_grid = param_grid, cv = 3, n_jobs = n_jobs, verbose=0)
        gd.fit(x_train, init_y.ravel())
        RF_paras = gd.best_params_
        RF_paras = [RF_paras['max_depth'], RF_paras['min_samples_leaf']]
            
    max_depth, min_samples_leaf = RF_paras
    rseed(0); npseed(0)
    Q = RF(max_depth = max_depth, random_state = 0, n_estimators = n_trees, min_samples_leaf =   
           min_samples_leaf, n_jobs = n_jobs, verbose = 0) 
    Q.fit(x_train, init_y.ravel())

    y_old = init_y.copy()
    # evaluate the policy policy
    rep, epsilon = 0, 100
    while(epsilon > threshold and rep < 100):
        rseed(0); npseed(0)
        y_train = UpdatedValues_eval(ss, policy, J, r, Q, gamma) # too slow [?]
        y_train = np.round(y_train, 6)
        epsilon = change_rate( y_old = y_old, y_new = y_train)
        Q = RF(max_depth = max_depth, random_state = 0, n_estimators = n_trees, min_samples_leaf =   
           min_samples_leaf, n_jobs = n_jobs, verbose = 0)
        Q.fit(x_train, y_train.ravel())  # regression function: (s,a) -> v

        y_old = y_train.copy()
        rep += 1
            
    def V_func(s):
        dx = s.shape[1]
        a = policy(s[:,(dx - (4 * J - 1)):dx]).reshape(-1,1)
        return Q.predict(np.hstack([s,a]))
    
    return V_func



