#%% packages
#%% 
from ._QRF import *
from ._uti_basic import *
from ._utility import *
#############################################################################
#############################################################################

def list2Matrix(List):
    # return a n * 1 matrix
    return np.array(np.expand_dims(np.array(List),1))

#%%

def TIGER_dynamics(state, action):
    p_correct = 0.7 # larger -> more POMDP
    # obs -> action -> obs, reward
    if action == 0: # listen
        p = rbin(1, p_correct)
        obs = p * state +  (1-p) * (0-state)
        reward = -1
    else: # action = -1 or 1
        if action == state:
            reward = -100
        else: # no tiger door
            reward = 10
        obs = 3 # end status
    return reward, obs


def TIGER_choose_action(obs, behav_def = 0):
    """
    behav_def:
        0. always listen
        1. random
        2. adaptive
    """
    p_listen = 0.9 # for random policy
    T_must_obs = 10 # for adaptive plicy
    
    if behav_def == 0:
        return 0 # always listen
    elif behav_def == 1:
        if rbin(1, p_listen):
            return 0
        elif rbin(1, 0.5):
            return 1
        else:
            return -1
    elif behav_def == 2:
        """ based on obs, Chengchun's approach
        1. if n <= T_must_obs: obs
        2. else: n > T_must_obs 时 p_listen = (1- max(p_left,p_right)) * 2, o.w. open the door accourding to the prob.
        """
        if obs[1] <= T_must_obs:
            return 0
        else:
            p_l = obs[0]
            p_listen = (1- max(p_l,1 - p_l)) * 2
            if rbin(1, p_listen):
                return 0
            elif rbin(1, p_l):
                return -1
            else:
                return 1
        
def simu_tiger(N = 1, T = 20, seed = 1, behav_def = 0, obs_def = "alt", T_def = 0, include_reward = True, fixed_state_comp = False):
    """
    T: spycify the game here
    A: "listen"/ "open_l" / "open_r"  ---- 0 / -1 / +1
    State:  "l" / "r" : -1 / +1
    Obervation: hear "l" / "r"
    Reward: -1, 10, - 100
    Returns: a list (len = N) of [$O_{T*dim_O},A_{T*1}$] or [O,A,R]
    
    behav_def:
        0. always listen
        1. random
        2. adaptive
    obs_def:
        "alt": [1,-1]
        1: [p]
        2: [p,n]
    T_def:
        0: length = T with always listen
        1: truncation
    """   
    # gamma = .9 
    
    MDPs = []
    rseed(seed); npseed(seed)
    init_state = rbin(1, .5, N) * 2 - 1
    true_states = []
    
    if T_def == 1:
        def stop(obs,t):
            return obs != 3
    else:
        def stop(obs,t):
            return t < T
    
    for i in range(N):
        ## Initialization
        state = init_state[i]
        obs, obs_hist = 0, [0]
        A = []
        R = [0] # for alignment purpose
        O, O_1 = [[0.5, 0]], [0.5]  
        t, left_cnt = 0, 0
        
        while(stop(obs,t)): # not in the Terminal state
            ## choose actiom, receive reward and state trainsition [observations]
            action = TIGER_choose_action(obs = O[-1], behav_def = behav_def) # obs = [p,n], old version
            reward, obs = TIGER_dynamics(state,action)
            
            ## record
            left_cnt += (obs == -1)
            t += 1
            # for obs_def_0
            obs_hist.append(obs)
            # for obs_def_1
            O_1.append(left_cnt/t)
            # for action choosing and obs_def_2
            if obs == 3:
                O.append([left_cnt/(t-1),t])
            else:
                O.append([left_cnt/t,t])  
            A.append(action)
            R.append(reward)
        A.append(3)
        
        if obs_def == "alt":
            O =  list2Matrix(obs_hist)
        elif obs_def == "null":
#             O =  list2Matrix(obs_hist)        
            if fixed_state_comp:
                O = list2Matrix(obs_hist)
                true_states.append(state)
            else:
                O = np.array([[a,state] for a in obs_hist])
#             print(O.shape)
        elif obs_def == 1:
            O = list2Matrix(O_1)
        elif obs_def == 2:
            O = np.array(O)
        if include_reward:
            MDP = [O, list2Matrix(A), list2Matrix(R)]
        else:
            MDP = [O, list2Matrix(A)]
        MDPs.append(MDP)
    if fixed_state_comp:
        return [MDPs,true_states]
    return MDPs

