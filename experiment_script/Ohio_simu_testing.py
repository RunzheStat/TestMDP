# -*- coding: utf-8 -*-

import os, sys
package_path = os.path.dirname(os.path.abspath(os.getcwd()))

sys.path.insert(0, package_path + "/test_func")
from _core_test_fun import *

sys.path.insert(0, package_path + "/experiment_func")
from _DGP_Ohio import *

os.environ["OMP_NUM_THREADS"] = "1"
#####################################
# To reduce computational cost, in our experiment, we use the “CV_once” option, which means we only do cross-validation in the 1st replication, 
# and use the chosen parameters in the remaining replications. With small-scale experiments, 
# the difference with standard cross-validation is negligible and will not affect our findings.
#####################################

def one_time(seed = 1, J = 1, J_upper = 10,
                     N = 30, T = 2 * 24, B = 200, Q = 10, sd_G = 5,
                     gamma_NFQ = 0.95, 
                     T_eval = 60, N_eval = 100, gamma_eval = 0.9, thre_eval = 1e-4,
                     paras = "CV", n_trees = 200,
                     first_T = 10, 
                     do_eval = True):
    ## generate data
    data = simu_Ohio(T, N, seed = seed, sd_G = sd_G)
    data = burn_in(data,first_T)
    T -= first_T
    # for value evaluation [we will use the original transition], 
    # do not use normalized data[will not be dominated like testing]
    value_data = data
    testing_data = [a[:2] for a in normalize(data)]
    ## this one time is used to get paras
    if paras == "CV_once": 
        return lam_est(data = testing_data, J = J, B = B, Q = Q, paras = paras, n_trees = n_trees)
    time = now()
    p_value = test(data = testing_data, J = J, B = B, Q = Q, paras = paras, n_trees = n_trees, print_time = False, method = "QRF")
    if seed % 100 == 0:
        print("** testing time:", now() - time, " for seed = ", seed,"**"); time = now()
    
    if do_eval: # for the currect J, get data, learn a function, and evaluate via simulations
        Learning_PatternSets = MDP2Trans(MDPs = value_data, J = J, action_in_states = True)
        Q_func = NFQ(PatternSets = Learning_PatternSets, gamma = gamma_NFQ,
                     RF_paras = paras, n_trees = n_trees, threshold = thre_eval)
        J_values = eval_Ohio_policy(Q_func = Q_func, J_Q = J, J_upper = J_upper,
                                   T = T_eval, gamma = gamma_eval, N = N_eval, 
                                   sd_G = sd_G, seed = 0)
        return [p_value, np.mean(J_values)] 
    else:
        return p_value


def one_setting_one_J(rep_times = 2, J = 1, J_upper = 5,
                      N = 20, T = 2 * 24, B = 2, Q = 10, sd_G = 5,
                      paras = "CV_once", n_trees = 20, 
                      init_seed = 0, do_eval = False, parallel = False, email = False):
    a = now()
    if paras == "CV_once":
        paras = one_time(seed = 0, J = J, J_upper = J_upper,
                         N = N, T = T, B = B, Q = Q,
                         sd_G = sd_G, 
                         paras = "CV_once", n_trees = n_trees,
                         do_eval = do_eval)   
        print("CV paras:",paras)
    
    def one_test(seed):
        return one_time(seed = seed, J = J, J_upper = J_upper,
                     N = N, T = T, B = B, Q = Q, sd_G = sd_G,
                     paras = paras, n_trees = n_trees, 
                     do_eval = do_eval)            
    if parallel:
        if rep_times == 500 and do_eval:
            r = []
            for i in range(5): # connection
                r_i = parmap(one_test, range(init_seed + i * 100, init_seed + (i + 1) * 100), parallel)
                print("the first", (i + 1) * 100, "reps in 500 reps Done: \n",
                      rej_rate([a[0] for a in r_i], [.1,.05,.01]), 
                     "\n with time cost: \n", now() - a)
                r += r_i
        else:
            r = parmap(one_test, range(init_seed, init_seed + rep_times), parallel)
    else:
        r = rep_seeds_print(one_test,rep_times,init_seed)
    print("total testing time cost for one J:", np.round(now() - a,3),Dash)
    if do_eval:
        p_values = [a[0] for a in r]
        rej_rates = rej_rate(p_values, [.1,.05,.01])
        values = [a[1] for a in r]
        if email:
            send_email("J = " + str(J) + "with testing results: \n" + str(rej_rates) + \
                      "\n and values: \n" + str([np.mean(values), np.std(values)]))
        return rej_rates, np.round(np.mean(values),4), np.round(np.std(values),4)
    else:
        rej_rates = rej_rate(r, [.1,.05,.01])
        return rej_rates

def one_setting_mul_J(rep_times = 50, N = 30, T = 24 * 2, B = 200, Q = 10, sd_G = 5, 
                      paras = "CV_once", n_trees = 200, 
                      init_seed = 0,
                      file = None, email = False, J_low = 1, J_upper = 5, 
                      do_eval = True, parallel = False, print_every_J = False):
    J_rej_rates, J_values = [], []
    true_lag = 4
    ## Prepare log
    setting = [N, T, sd_G]
    email_contents = ""
    email_setting = [rep_times, N, T, B, sd_G]
    email_str = "rep_times, N, T, B, sd_G"
    print(dash, "Setting running [N,T,sd_G]:", setting, dash, "\n")
    if file is not None:
        print(dash, "Setting running [N,T,sd_G]:", setting, dash, "\n", file = file)
    ## Testing and value results for each J, with one true_lag
    for J in range(J_low, J_upper + 1):
        a = now()
        rej_rates, mean_value, std_value = one_setting_one_J(rep_times = rep_times, J = J, J_upper = J_upper,
                                                 N = N, T = T, B = B,Q=Q, sd_G = sd_G, 
                                                 paras = paras, n_trees = n_trees, 
                                                 init_seed = init_seed, 
                                                 do_eval = do_eval, parallel = parallel)

        #### Store results
        J_rej_rates.append(rej_rates)
        J_values.append([mean_value, std_value]) # sd_over_500(mean_over_10)
        
        #### Prepare log
        print_res = ' '.join(["\n", "Above: when true_lag = ",str(true_lag),
              "and we do J = ", str(J), "testing",str(is_null(true_lag = true_lag, J = J)), 
              "[supremum-based, integration-based]", "\n The average and std of values: \n",
                              str([mean_value, std_value])])
        print(print_res)
        if file is not None:
            print(print_res, file = file)
        
        print_time = ' '.join(["Time cost:", str(np.round( (now() - a)/60,2)), "mins","\n",DASH])
        print(print_time)
        if file is not None:
            print(print_time, file = file)
        
        log = "12_16, lag4 OLS AWS" + ", init_seed - " + str(init_seed) + "\n" + dash + "\n"
        email_this_J = email_str + "\n" + str(email_setting)+ '; J=' + str(J) + ' DONE!\n' \
            +'alpha = [.1,.05,.01]' + ', [supremum-based, integration-based] \n' + str(rej_rates) + "\n" \
            + str([mean_value, std_value]) + "\n"
        email_contents += dash + "\n" + email_this_J
        if print_every_J:
            print(J_rej_rates, DASH, J_values)
    ## Final printing out
    if email:
        send_email(log + email_contents)
    if  file is not None: # print latex
        latex_ohio_one_T_sd_G_mul_j(J_rej_rates,file)
        
    return J_rej_rates, J_values

print("Import DONE!")


rr = []
for N in [10, 15, 20]:
    r = one_setting_mul_J(rep_times = 500, N = N, T = 7 * 8 * 24, sd_G = 3,
                  paras = "CV_once", n_trees = 100, 
                  B = 100, init_seed = 0, 
                  J_low = 1, J_upper = 10, 
                  do_eval = False, parallel = n_cores, print_every_J = True)
    print(r)
    rr.append(r)
    print(rr)
