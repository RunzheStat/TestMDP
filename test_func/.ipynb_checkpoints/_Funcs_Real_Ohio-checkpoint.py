#############################################################################
#%%
from ._utility import * 
from ._QRF import * 
from ._core_test_fun import *
from ._DGP_Ohio import *
from ._uti_basic import * 
from ._utility_RL import * 
os.environ["OMP_NUM_THREADS"] = "1"
n_cores = multiprocessing.cpu_count()
#############################################################################
#%% Data
import sys, os
path = os.getcwd() + "/code_data/Data_Ohio.csv"
data = pd.read_csv(path, header = 0)
data0 = np.array(data)
data0 = data0[:,1:] # no row index
#############################################################################

def generate_initial_states(N_init = 100, J_upper = 5, seed = 0):
    """generate initial states for comparison of values in the Ohio real data.
    """
    init_obs, init_A = simu_Ohio(T = J_upper, N = N_init, 
                                 seed = seed, sd_G = 3, matrix_output = True, is_real = True)
    init_A = init_A.reshape(1, J_upper, N_init)
    initial_states = np.concatenate([init_obs, init_A], 0)
    initial_states = initial_states.reshape((4 * J_upper, N_init), order = "F")
    initial_states = initial_states[:(J_upper * 4 - 1), :] 
    return initial_states.T
    
    
def process_data_Nsplit(index, T):
    """
    data: a list(len-N) of [T*3 states, T * 1 actions T * 1 rewards]
    T: length = 1100
    """
    data, J_data = [], []
    for i in index:
        temp = data0[T * i : T * (i+1)].copy()
        states = temp[:, :3]
        actions = temp[:, 3].reshape(-1, 1)
        rewards = np.roll(apply_v(Glucose2Reward, states[:, 0]), shift = -1).reshape(-1, 1) 
        J_data.append([states, actions])
        data.append([states, actions, rewards])
    return data.copy(), normalize(J_data.copy())


#############################################################################
#############################################################################
#%%
 
def real_ohio_Nsplit(J_upper = 10, gamma = 0.9, gamma_eval = 0.9, alpha = 0.02, RF_paras = "CV", n_trees = 100, 
                     N_init = 100, seed = 1, n_set = 20,
                     parallel = False, T_unify = 1100, threshold = 1e-4):
    a = now()
    init_states = generate_initial_states(N_init = N_init, J_upper = J_upper, seed = seed) # N * (J_upper * 4 - 1)
        
    arr = [i for i in range(6)]
    rseed(seed); npseed(seed)
    all_possible_train_set = permutation(list(combinations(arr, 3)) )
    def one_time(i):
        time = now()
        train_set = all_possible_train_set[i]
        eval_set = set([i for i in range(6)]) - set(train_set)   
        train_data, J_data = process_data_Nsplit(train_set, T = T_unify)
        eval_data, _ = process_data_Nsplit(eval_set, T = T_unify) 
        
        ### Given a J, get the optimal policy and evaluate its performance
        eval_PatternSets = MDP2Trans(MDPs = eval_data, J = J_upper, action_in_states = True)
        values = []        
        for J in range(1, J_upper + 1):
            ### Learn the optimal policies
            Learning_PatternSets = MDP2Trans(MDPs = train_data, J = J, action_in_states = True)
            Q_func = NFQ(PatternSets = Learning_PatternSets, gamma = gamma,
                             RF_paras = RF_paras, n_trees = n_trees, threshold = threshold)
            ### Evaluate the policy: learned Q and observed trajectories
            V_func = FQE(PatternSets = eval_PatternSets, Q_func = Q_func, J = J,
                                     gamma = gamma_eval, RF_paras = RF_paras, n_trees = n_trees,
                                          threshold = threshold)
            
            ### Evaluate using init states
            values_integration = V_func(init_states)
            value = np.round(np.mean(values_integration), 4)
            values.append(value)
        ### Store results
        print("The ", i + 1, "round ends with Time cost:", np.round(now() - time,2), "\n")
        
        return values
    
    r_values = parmap(one_time, range(n_set))
    r_values = np.array(r_values)

    print("mean:", np.mean(r_values, 0), "\n", "std:", np.std(r_values, 0))
    print("time cost: ", now() - a)
    return r_values

#############################################################################
#%% Decide the order with all data

def decide_J(data, J_range, paras = "CV", n_trees = 100, T = 1100):
    data_J = []
    for i in range(6):
        temp = data[T * i : T * (i + 1)]
        temp = [temp[:, :3], temp[:, 3].reshape(-1, 1)]
        data_J.append(temp)
    data_J = normalize(data_J)
    r = selectOrder(data_J, B = 200, Q = 10, L = 3, alpha = 0.1, K = 10, paras="CV", n_trees = n_trees)
    return r

# def decide_J(data, J_range, paras = "CV", n_trees = 100, T = 1100):
#     data_J = []
#     for i in range(6):
#         temp = data[T * i : T * (i + 1)]
#         temp = [temp[:, :3], temp[:, 3].reshape(-1, 1)]
#         data_J.append(temp)
#     data_J = normalize(data_J)
#     def test_one_J(J):
#         return test(data_J, J = J, B = 200, Q = 10, paras = paras, n_trees = n_trees)
#     r = parmap(test_one_J, J_range, n_cores)
#     print(r)
#     return r

