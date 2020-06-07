# -*- coding: utf-8 -*-

import os, sys
package_path = os.path.dirname(os.path.abspath(os.getcwd()))

sys.path.insert(0, package_path + "/test_func")
from _core_test_fun import *

sys.path.insert(0, package_path + "/experiment_func")
from _DGP_Ohio import *
from _utility_RL import *

os.environ["OMP_NUM_THREADS"] = "1"
#####################################
# To reduce computational cost, in our experiment, we use the “CV_once” option, which means we only do cross-validation in the 1st replication, 
# and use the chosen parameters in the remaining replications. With small-scale experiments, 
# the difference with standard cross-validation is negligible and will not affect our findings.
#####################################
def one_time_value_only(seed = 1, J = 1, J_upper = 10,
                     N = 10, T = 56 * 24, B = 100, Q = 10, sd_G = 3,
                     gamma_NFQ = 0.9, 
                     T_eval = 60, N_eval = 100, gamma_eval = 0.9, thre_eval = 1e-4,
                     paras = "CV_once", n_trees = 100,
                     first_T = 10, true_lag = 4):
    ## generate data
    a = now()
    data = simu_Ohio(T, N, seed = seed, sd_G = sd_G)
    data = burn_in(data,first_T)
    T -= first_T
    value_data = data
    testing_data = [a[:2] for a in normalize(data)]
    ## this one time is used to get paras
    if paras == "CV_once": 
        return lam_est(data = testing_data, J = J, B = B, Q = Q, paras = paras, n_trees = n_trees)
    time = now()
    Learning_PatternSets = MDP2Trans(MDPs = value_data, J = J, action_in_states = True)
    Q_func = NFQ(PatternSets = Learning_PatternSets, gamma = gamma_NFQ,
                 RF_paras = paras, n_trees = n_trees, threshold = thre_eval)
    if seed % 100 == 0:
        print("** Learning [for value] time cost:", np.round(now() - time, 3) , "**"); time = now()
    J_values = eval_Ohio_policy(Q_func = Q_func, J_Q = J, J_upper = J_upper,
                               T = T_eval, gamma = gamma_eval, N = N_eval, 
                               sd_G = sd_G, seed = 0)#, true_lag = 4)
    return np.mean(J_values)


def one_setting_one_J_value_only(rep_times = 500, J = 1, J_upper = 10,
                      N = 10, T = 56 * 24, B = 100, Q = 10, sd_G = 3,
                      paras = "CV_once", n_trees = 100, 
                      gamma_NFQ = 0.9, 
                      T_eval = 60, N_eval = 100, gamma_eval = 0.9, thre_eval = 1e-4,
                      parallel = False, path = None):
    a = now()
    if paras == "CV_once":
        paras = one_time_value_only(seed = 0, J = J, J_upper = J_upper,
                         N = N, T = T, B = B, Q = Q,
                         sd_G = sd_G, gamma_NFQ = gamma_NFQ,
                         T_eval = T_eval, N_eval = N_eval, 
                      gamma_eval = gamma_eval, thre_eval = thre_eval,
                         paras = "CV_once", n_trees = n_trees)   
    def one_time(seed):
        return one_time_value_only(seed = seed, J = J, J_upper = J_upper,
                     N = N, T = T, B = B, Q = Q, sd_G = sd_G,
                     gamma_NFQ = gamma_NFQ, 
                     T_eval = T_eval, N_eval = N_eval, 
                      gamma_eval = gamma_eval, thre_eval = thre_eval,
                     paras = paras, n_trees = n_trees)            

    values = parmap(one_time, range(rep_times), parallel)

    print("total time cost for one J:", np.round(now() - a, 3), Dash)
    
    u_val = np.round(np.mean(values),4)
    sd_val = np.round(np.std(values),4)
    
    return values, u_val, sd_val



def one_setting_value_only(rep_times = 500,
                      N = 10, T = 56 * 24, B = 100, Q = 10, sd_G = 3,
                      paras = "CV_once", n_trees = 100, 
                      gamma_NFQ = 0.9, 
                      T_eval = 60, N_eval = 100, gamma_eval = 0.9, thre_eval = 1e-4,
                      parallel = False, file = None):
    rr = []
    value_details = []
    for J in range(1, 11):
        r = one_setting_one_J_value_only(rep_times = rep_times, J = J, J_upper = 10,
                      N = N, T = T, B = B, Q = Q, sd_G = sd_G,
                      paras = paras, n_trees = n_trees, 
                      gamma_NFQ = gamma_NFQ, 
                      T_eval = T_eval, N_eval = N_eval, 
                      gamma_eval = gamma_eval, thre_eval = thre_eval,
                      parallel = parallel)
        rr.append([r[1], r[2]])
        value_details.append(r[0])
        print("the currect results for J = ", J, ":\n", rr, DASH)
    print_content = "N = " + str(N) + "sd = " + str(sd_G) + ":" + str(rr)
    print(print_content, file = file)
    return rr, value_details

print("import DONE!", "num of cores:", n_cores, DASH)


#%% Time Cost


path = "Ohio_simu_values.txt" # 0128 reruned and reproduced
file = open(path, 'w')
reps = 500
gamma = 0.9
T_eval = 60
sd_G = 3
value_details = []
mean_values = []

for N in [10, 15, 20]: 
    print(DASH, "[N, sd_G] = ", [N, sd_G], DASH)
    r, value_detail = one_setting_value_only(rep_times = reps,
                      N = N, T = 8 * 7 * 24, B = 100, Q = 10, sd_G = sd_G,
                      paras = "CV_once", n_trees = 100, 
                      gamma_NFQ = gamma,
                      T_eval = 60, N_eval = 100, 
                      gamma_eval = gamma, thre_eval = 1e-4,
                      parallel = n_cores, file = file)
    print(DASH, "[N, sd_G] = ", [N, sd_G], "r:", r, DASH)
    mean_values.append(r)
    value_details.append(value_detail)
file.close()

res = [mean_values, value_details]
with open("Ohio_simu_value.list", 'wb') as file:
    pickle.dump(res, file)
file.close()