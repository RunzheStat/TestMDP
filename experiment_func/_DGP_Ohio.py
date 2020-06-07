#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% packages

################################################################################################
import os, sys
package_path = os.path.dirname(os.path.abspath(os.getcwd()))

sys.path.insert(0, package_path + "/test_func")
from _core_test_fun import *
from _utility_RL import *

################################################ OHIO ##########################################
################################################################################################
# the following parameters will not change with the LM fitting
const = 39.03
init_u_G = 162  
init_sd_G = 60
p_D, u_D, sd_D = 0.17, 44.4, 35.5
p_E, u_E, sd_E = 0.05, 4.9, 1.04
p_A = [0.805, 0.084, 0.072, 0.029, 0.010] # new discritization
range_a = [0, 1, 2, 3, 4]

##########################################
# left to right: t-4, .. , t-1
coefficients = [-0.008     ,  0.106     , -0.481     ,  1.171     ,  # glucose
          0.008     ,  -0.004     ,  0.08      ,  0.23      ,  # diet
          0.009     , -1.542     , 3.097     , -3.489     ,  # exercise
          -0.30402253, -2.02343638, -0.3310525 , -0.43941028] # action

def Glucose2Reward(gl, definition = 1):
    # Q: too sensitive?
    low_gl = 80
    high_gl = 140
    return np.select([gl>=high_gl, gl<=low_gl, low_gl<gl<high_gl], [-(gl-high_gl)**1.35/30, -(low_gl-gl)**2/30, 0])

################################################################################################
################################################################################################

def init_MDPs(T, N, sd_G = 3, seed = 0): 
    """
    Randomly initialize 
        1. G_t [0,..., T_true_lag];
        2. errors for G_t
        3. when to take how many diets/exercises [matters?]
    Outputs:
        init G_t and its future erroes; all D_t and E_t
    """
    rseed(seed); npseed(seed)
    true_lag = 4
    obs = np.zeros((3, T, N))  # [Gi, D, Ex]
    obs[0, :true_lag, :] = rnorm(init_u_G, init_sd_G, true_lag * N).reshape(true_lag, N)
    
    e_D = abs(rnorm(u_D, sd_D, T * N))
    e_E = abs(rnorm(u_E, sd_E, T * N))
    obs[1, :, :] = (rbin(1, p_D, T * N) * e_D).reshape((T, N))
    obs[2, :, :] = (rbin(1, p_E, T * N) * e_E).reshape((T, N))
    
    e_G = rnorm(0, sd_G, T * N).reshape((T, N))
    
    return obs, e_G



def useful_obs(obs, actions, t):
    true_lag = 4
    r = np.vstack([
        obs[0, (t - true_lag):t, :], obs[1, (t - true_lag):t, :],
        obs[2, (t - true_lag):t, :], actions[(t - true_lag):t, :]])
    return r

def next_obs(tran_mat, useful_last_obs, e_G, t):
    return np.array(const).reshape((1, 1)) + tran_mat.dot(useful_last_obs) + np.array([e_G[t, :]])


################################################################################################
################################################################################################

def simu_Ohio(T = 5, N = 2, seed = 1, sd_G = 5.5, matrix_output = False, is_real = False):
    """ Simulate N patient trajectories with length T, calibrated from the Ohio dataset.
    """
    tran_mat = np.expand_dims(arr(coefficients), 0)

    # Initialization
    if is_real:
        obs, e_G = init_MDPs_real(T = T, N = N, sd_G = sd_G, seed = seed)
    else:
        obs, e_G = init_MDPs(T = T, N = N, sd_G = sd_G, seed = seed)
    rseed(seed); npseed(seed)
    actions = np.random.choice(range(len(p_A)), size=T * N, p=p_A).reshape((T, N))
    # Transition
    for t in range(4, T):
        useful_last_obs = useful_obs(obs = obs, actions = actions, t = t)
        obs[0, t, :] = next_obs(tran_mat = tran_mat, useful_last_obs = useful_last_obs, e_G = e_G, t = t)
    
    if matrix_output: # for eval_Ohio_policy below
        return obs, actions
    # Collection
    MDPs = []
    for i in range(N):
        MDPs.append([obs[:, :, i],
                     actions[:, i]])
    s_a = [[a[0].T, np.array(a[1]).reshape(T, 1), ] for a in MDPs]
    MDPs = [[a[0], a[1], np.roll(apply_v(Glucose2Reward, a[0][:, 0].reshape(-1, 1)), shift = -1).reshape(-1, 1)] 
            for a in s_a]
    return  MDPs # a list of [obs = T * 3, actions, rewards]

################################################################################################
# Funcs for Simu Ohio Values
################################################################################################

def eval_Ohio_policy(Q_func, J_Q, T, N, J_upper, sd_G = 0, gamma = 0.9, debug = 0, seed = 0):
    """ Evaluate the value of a policy in simulation.
    
    Randomly the first four time points，
    and then follow the simulation model until T = 10, 
    and then begin to use policy and collect rewards:
    
    1. choosing actions following Q, 
    2. trans following the environment
    3. collecting rewards.
    """
    policy = Estpolicy(Q_func, range_a)
    tran_mat = np.expand_dims(arr(coefficients), 0)
    
    ### Initialize the first 10
    true_lag = 4
    init_obs, init_A = simu_Ohio(T = 10, N = N, seed = 0, sd_G = sd_G, matrix_output = True)
    obs, e_G = init_MDPs(T = T, N = N, sd_G = sd_G, seed = seed)
    obs[:, :10, :] = init_obs
    actions = np.zeros((T, N)) # store previous actions
    actions[:10, :] = init_A
    
    rseed(seed); npseed(seed)
    dim_obs = obs.shape[0]
    for t in range(J_upper, T): 
        # next observations: based on ...,t-1, to decide t.
        useful_last_obs = useful_obs(obs = obs, actions = actions, t = t)
        obs[0, t, :] = next_obs(tran_mat = tran_mat, useful_last_obs = useful_last_obs, e_G = e_G, t = t)
        
        # choose actions based on status. obs = [3, T, N]      
        s = ObsAct2State(obs, actions, t, J_Q, multiple_N = True) # dim * N
        A_t = policy(s.T).T # s [N * dx] -> actions [N * 1] -> 1 * N
        actions[t, :] = A_t
                
    # collect rewards
    Values = est_values(obs, gamma = gamma, init_T = J_upper)
    return Values

def est_values(obs, gamma = 0.9, init_T = 10):
    """ Tool to calculate culmulative rewards from observation (glucose histroy)
    Input: the observed trajectories (possibly based on the optimal policy)
    3 * T * N
    Output: the collected culmulative rewards
    
    init_T: when the glucose becomes stable
    """
    Values = []
    N = obs.shape[2]
    T = obs.shape[1]
    for i in range(N):
        rewards = np.roll(apply_v(Glucose2Reward, obs[0, init_T:, i]), shift = -1).reshape(-1, 1)
        est_Value = np.round(cum_r(rewards, gamma), 3)
        Values.append(est_Value[0])
    return Values


def init_MDPs_real(T, N, sd_G, seed = 0): # version of new -> bad simu results
    """
    Randomly initialize 
        1. G_t [0,..., T_true_lag];
        2. errors for G_t
        3. when to take how many diets/exercises [matters?]
    Outputs:
        init G_t and its future erroes; all D_t and E_t
    """
    rseed(seed); npseed(seed)
    true_lag = 4
    obs = np.zeros((3, T, N))  # [Gi, D, Ex]
    e_D = abs(rnorm(u_D, sd_D, T * N))
    e_E = abs(rnorm(u_E, sd_E, T * N))
    e_G = rnorm(0, sd_G, T * N).reshape((T, N))
    
    obs[0, :true_lag, :] = rnorm(init_u_G, init_sd_G, true_lag * N).reshape(true_lag, N)
    obs[1, :, :] = (rbin(1, p_D, T * N) * e_D).reshape((T, N))
    obs[2, :, :] = (rbin(1, p_E, T * N) * e_E).reshape((T, N))
    
    return obs, e_G