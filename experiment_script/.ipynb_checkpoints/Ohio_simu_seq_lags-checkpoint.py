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

def one_time_seq(seed = 1, J_upper = 10, alpha_range = [0.02, 0.01, 0.005], 
                     N = 10, T = 7 * 8 * 24, B = 100, Q = 10, sd_G = 3,
                     para_ranges = None, n_trees = 100, 
                     ):
    ## generate data
    data = simu_Ohio(T, N, seed = seed, sd_G = sd_G)
    data = burn_in(data, first_T = 10)
    T -= 10
    # for value evaluation [we will use the original transition], 
    # do not use normalized data[will not be dominated like testing]
    value_data = data
    testing_data = [a[:2] for a in normalize(data)]    
    time = now()
    p_values = []
    for J in range(1, J_upper + 1):
        p_value = test(data = testing_data, J = J, B = B, Q = Q, paras = para_ranges[J - 1], 
                       n_trees = n_trees, print_time = False, method = "QRF")
        p_values.append(p_value)
        if p_value > alpha_range[0]: 
            break
    lags = []
    for alpha in alpha_range:
        for i in range(J_upper):
            if p_values[i] > alpha:
                lags.append(i + 1)
                break
        if i == J_upper - 1: 
            lags.append(J_upper)
            
    if seed % 50 == 0:
        print("** testing time:", now() - time, " for seed = ", seed,"**"); time = now()
    return [lags, p_values]

def one_setting_seq(rep_times = 500, N = 10, T = 24 * 56, B = 100, Q = 10, sd_G = 3, 
                      n_trees = 100, alpha_range = [0.02, 0.01, 0.005], 
                      init_seed = 0,
                      file = None, J_low = 1, J_upper = 10, 
                      parallel = 10):
    # CV_paras for each J
    para_ranges = []
    data = simu_Ohio(T, N, seed = 0, sd_G = sd_G)
    data = burn_in(data, first_T = 10)
    T -= 10
    testing_data = [a[:2] for a in normalize(data)]
    for J in range(1, J_upper + 1):
        paras = lam_est(data = testing_data, J = J, B = B, Q = Q, paras = "CV_once", n_trees = n_trees, method = "QRF")
        para_ranges.append(paras)
    def one_time(seed):
        r = one_time_seq(seed = seed, J_upper = J_upper, alpha_range = alpha_range, 
                     N = N, T = T, B = B, Q = Q, sd_G = sd_G,
                     para_ranges = para_ranges, n_trees = n_trees)
        if seed % 50 == 0:
            print(seed, "Done!\n")
        return r
    
    r = parmap(one_time, range(init_seed, init_seed + rep_times), parallel)
    # different alphas
    lagss, p_valuess = [a[0] for a in r], [a[1] for a in r]
    print(lagss, DASH, DASH, p_valuess, DASH)
    lags_each_alpha = []
    for i in range(len(alpha_range)):
        lags_each_alpha.append([a[i] for a in lagss])
    r = [lags_each_alpha, p_valuess]
    if file is not None:
        print(DASH + str([N, sd_G]), file = file)
        for i in range(4):
            print(str(r[i]) + dasH, file = file)
    return r
print("import DONE!", "num of cores:", n_cores, DASH)

#####################################
path = "Ohio_seq.txt"
file = open(path, 'w')
rr = []
times = 500
sd_G = 3
for N in [10, 15, 20]:
    print([N, sd_G],": \n")
    r = one_setting_seq(rep_times = times, N = N, T = 24 * 7 * 8, sd_G = 3, 
                  n_trees = 100, B = 100, Q = 10, alpha_range = [0.01, 0.005], 
                  init_seed = 0, 
                  file = file, J_low = 1, J_upper = 10, 
                  parallel = n_cores)
    rr.append(r)
file.close()


with open("Ohio_seq.list", 'wb') as file:
    pickle.dump(rr, file)
file.close()
