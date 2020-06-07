#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
#############################################################################
#%% packages
import numpy as np
import scipy as sp
from scipy.linalg import sqrtm
import pandas as pd
from numpy import absolute as np_abs
from random import seed as rseed
from numpy.random import seed as npseed
from numpy.random import normal as rnorm
from numpy.random import uniform as runi
from numpy.random import binomial as rbin
from numpy.random import shuffle,randn, permutation # randn(d1,d2) is d1*d2 i.i.d N(0,1)
from numpy import array as arr
from numpy import sqrt, cos, sin, exp, dot, diag, quantile, zeros, roll, multiply, stack, concatenate
from numpy import concatenate as v_add
from numpy.linalg import norm
from numpy import apply_along_axis as apply
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import GridSearchCV
from itertools import combinations 
import operator
import time
now = time.time
from sklearn.ensemble import RandomForestRegressor as RandomForest
from sklearn.model_selection import KFold
from statsmodels.stats import proportion as prop
import os

#############################################################################
#############################################################################

#%% utility funs

def CI_prop(n,p):
    """ 
    Input: In n reps, observed proportion p
    Output: the 95% CI  of this p
    """
    r = prop.proportion_confint(n * p, n, alpha = 0.05, method='binom_test')
    return np.round([r[0], r[1]],4)

def normalize_unit_sd(array):
    def temp(v):
        return v / np.std(v)
    return np.array(apply(temp, 0, array))

def apply_v(f,v):
    return np.array([f(a) for a in v])

def burn_in(data,first_T):
    if len(data[0]) == 2:
        return [[patient[0][first_T:,:], patient[1][first_T:,:]] for patient in data]
    else:
        return [[patient[0][first_T:,:], patient[1][first_T:,:], patient[2][first_T:,:]] for patient in data]

flatten = lambda l: [item for sublist in l for item in sublist]

def is_null(true_lag,J):
    if J >= true_lag:
        return "(H0)"
    else:
        return "(H1)"

def list2Matrix(List):
    # return a n * 1 matrix
    return np.array(np.expand_dims(np.array(List),1))

def round_list(thelist,dec):
    """
    extend np.round to list
    """
    return [round(a,dec) for a in thelist]


def normalize(data, centralized = False):
    """
    normalize the simulated data
    data: len-n of [T*dx,T*da]
    Returns: data
    """
    state, action = [a[0].copy() for a in data], [a[1].copy() for a in data]
    n = len(data)
    dx = state[0].shape[1]
    
    ### States
    for i in range(dx):
        s = np.array([a[:,i] for a in state])
        mean, sd = np.mean(s), np.std(s)
        for j in range(n):
            if centralized:
                state[j][:,i] -= mean
            if sd != 0:
                state[j][:,i] = state[j][:,i] / sd
                
    ### Action: 
    a = np.array(action)
    mean, sd = np.mean(a), np.std(a)
#     sd = 1
#     action = [ a / sd for a in action]

    ### Reward
    if len(data[0]) == 3:
        reward = [a[2] for a in data]
        a = np.array(reward)
        mean, sd = np.mean(a), np.std(a)
        if sd == 0:
            sd = 1
        if centralized:
            reward = [ (a - mean) / sd for a in reward]
        else:
            reward = [ a / sd for a in reward]
        return [[state[i],action[i],reward[i]] for i in range(n)]
    else:
        return [[state[i],action[i]] for i in range(n)]

#%% utility funs


def p_value(test_stat,sim_test_stats):
    """
    one testing result (p-value), Bootstrap-based.
    
    Default: the larger, the significant
    Return: p-value
    """
    return round(1 - sum(np.abs(test_stat) > np.abs(sim_test_stats)) / len(sim_test_stats),4)

def rej_rate(p_values, alphas):
    rep_times = len(p_values)
    p_values = np.array(p_values)
    RRs = []
    for alpha in alphas:
        RR = sum(p_values < alpha) / rep_times
        RRs.append(RR)
        print("Under alpha", alpha, "the rejection rate is:", RR)
    return RRs

def rej_rate_quite(p_values,alphas,file = None):
    rep_times = len(p_values)
    p_values = np.array(p_values)
    
    RRs = []
    for alpha in alphas:
        RR = sum(p_values < alpha) / rep_times
        RRs.append(RR)
    return RRs


def rej_rate_quick(p):
    r = []
    T = len(p)
    p = np.array(p)
    for i in [0.01,0.05,0.1]:
        r.append(np.sum( p < i) / T)
    return r

def rej_rate_seq(results):
    """
    Imput: a list (len = times) of [1,0]
    Output: [0.2,0.7]
    """
    results = np.array(results)
    times = results.shape[0]
    return np.sum(results,0) / times



def seq_rej_rate_mul_J(ps,alphas):
    """
    ps: len-J_upper list of np.array(times * 2)
    Output: if always rej, then rej
    """
    rej = []
    for alpha in alphas:
        aa = [np.array(p) < alpha for p in ps]
        bb = np.sum(np.array(aa), 0) == len(ps)
        rate = np.round(np.mean(bb, 0),3)
        rej.append(rate)
    return rej


#%%
def truncateMDP(MDPs,T):
    data = []
    l = len(MDPs[0])
    for MDP in MDPs:
        if (MDP[0].shape[0]) >= T:
            data.append([MDP[i][:T] for i in range(l)])
    return data


def p_sd(T):
    r = []
    for p_true in [0.01,0.05,0.1]:
        r.append(np.round(np.sqrt(p_true * (1 - p_true) / T),4))
    return r

def latex_ohio_one_T_sd_G_mul_j(a, file):
    for J in range(len(a)):
        print("J = ", J + 1, end = "    " , file = file)
        aa = a[J]
        for alpha in range(3):
            print(aa[alpha][0],"& ", end = "", file = file) # max
        print("\n", file = file)
        
def print_progress(i, N):
    if (i * 100 // N == 0):
        print("#", end = "", flush = True)