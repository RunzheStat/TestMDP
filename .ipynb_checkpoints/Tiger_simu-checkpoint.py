from code import *
import os
os.environ["OMP_NUM_THREADS"] = "1"
#####################################
# To reduce computational cost, in our experiment, we use the “CV_once” option, which means we only do cross-validation in the 1st replication, 
# and use the chosen parameters in the remaining replications. With small-scale experiments, 
# the difference with standard cross-validation is negligible and will not affect our findings.
#####################################
def one_time(seed = 1, J = 1, 
                     N = 100, T = 20, T_def = 0, 
                     B = 100, Q = 10, 
                     behav_def = 0, obs_def = "alt", 
                     paras = [100, 3,20], weighted = True, include_reward = False,
                     method = "QRF"):
    """
    include_reward: if include reward to our test
    T_def:
        0: length = T with always listen
        1: truncation
    T: the final length
    """
    ### generate data
    fixed_state_comp = (obs_def == "null")
    MDPs = simu_tiger(N = N, T = T, seed = seed,
                       behav_def = behav_def, obs_def = obs_def,
                       T_def = T_def, include_reward = include_reward, fixed_state_comp = fixed_state_comp)
    T += 1 # due to the DGP
    ### Preprocess
    if fixed_state_comp:
        MDPs, fixed_state_comp = MDPs
    else:
        fixed_state_comp = None
    if T_def == 1:
        MDPs = truncateMDP(MDPs,T)
    if not include_reward:
        MDPs = [a[:2] for a in MDPs]
    N = len(MDPs)
    ### Calculate
    if paras == "CV_once":
        return lam_est(data = MDPs, J = J, B = B, Q = Q, paras = paras, include_reward = include_reward,
                  fixed_state_comp = fixed_state_comp, method = method)
    return test(data = MDPs, J = J, B = B, Q = Q, paras = paras, #print_time = print_time,
                include_reward = include_reward, fixed_state_comp = fixed_state_comp, method = method)


def one_setting_one_J(rep_times = 10, J = 1, 
                      N = 100, T = 20, T_def = 0,
                      B = 100, Q = 10,
                      behav_def = 0, obs_def = "alt",
                      include_reward = False, mute = True,
                      paras = "CV_once", init_seed = 0, parallel = True, method = "QRF"):
    if paras == "CV_once":
        paras = one_time(seed = 0, J = J, 
                         N = N, T = T, B = B, Q = Q,
                        behav_def = behav_def, obs_def = obs_def,
                        paras = "CV_once",
                        T_def = T_def, include_reward = include_reward, method = method)
        print("CV paras:", paras)
    
    def one_test(seed):
        return one_time(seed = seed, J = J, 
                     N = N, T = T, B = B, Q = Q,
                    behav_def = behav_def, obs_def = obs_def,
                    T_def = T_def, include_reward = include_reward,
                     paras = paras, method = method)
    p_values = parmap(one_test,range(init_seed, init_seed + rep_times), parallel)
    if not mute:
        print("rejection rates are:", rej_rate_quick(p_values))
    return p_values
print("Import DONE!")

print("n_cores = ", n_cores)


for obs_def in ["null", "alt"]:
    for N in [50, 100, 200]:
        for J in range(1, 11):
            p_values = one_setting_one_J(rep_times = 500, J = J, 
                              N = N, T = 20, T_def = 0,
                              B = 100, Q = 10,
                              behav_def = 0, obs_def = obs_def,
                              include_reward = False, mute = False,
                              paras = "CV_once", init_seed = 0, parallel = n_cores, method = "QRF")
            rej_rate_quick(p_values)