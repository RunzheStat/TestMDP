#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is for the random forest-based method used in the paper "Does MDP Fit the Data?" to estimate conditional characteristic functions. 
The majority of functions in this file were adapted from the source code of the paper "Quantile Regression Forest" on Github.
    Date: 10/12/2019.
    URL:https://github.com/scikit-garden/scikit-garden/tree/master/skgarden
"""
##########################################################################
from  _uti_basic import *
##########################################################################
import warnings
warnings.filterwarnings('ignore')
from numpy.random import seed as rseed 
from numpy.random import randn # randn(d1,d2) is d1*d2 i.i.d N(0,1)
import numpy as np
from numpy import ma
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import time
now = time.time
##########################################################################

def weighted_est(y,uv,cos_sin,weights=None):
    """
    # weights: array-like, shape=(n_samples,)
    #    weights[i] is the weight given to point a[i] while computing the
    #    quantile. If weights[i] is zero, a[i] is simply ignored during the
    #    percentile computation.
    
    Parameters
    ----------
    # uv: assume is B * d_
    
    Returns
    -------
    B * 1, for a given T
    """
    if weights is None:
        return np.mean(cos_sin(y.dot(uv)),axis = 0)
    return weights.T.dot(cos_sin(y.dot(uv))) # v.T

def generate_sample_indices(random_state, n_samples):
    """
    [Just copied and pasted]
    Generates bootstrap indices for each tree fit.

    Parameters
    ----------
    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    n_samples: int
        Number of samples to generate from each tree.

    Returns
    -------
    sample_indices: array-like, shape=(n_samples), dtype=np.int32
        Sample indices.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices
##########################################################################   
# QRF <- QBF,QDT

class BaseForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=1)
        super(BaseForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like((self.y_train_leaves_), dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            bootstrap_indices = generate_sample_indices(est.random_state, len(y))
            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            y_train_leaves = est.y_train_leaves_
            for curr_leaf in np.unique(y_train_leaves):
                y_ind = y_train_leaves == curr_leaf
                self.y_weights_[i, y_ind] = (
                    est_weights[y_ind] / np.sum(est_weights[y_ind]))

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]
        return self
    
    def predict(self, X, uv=0): # , cos_sin
        """
        Predict cond. char. values for either forward or backward

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        uv: [B,dim_y]. can be either u or v
        Returns
        -------
        char_est : array of shape = [n_samples,B]
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc") # around N * T
        X_leaves = self.apply(X) # (n_test = N * T, n_tree) 
        weights = np.zeros((X.shape[0], len(self.y_train_)))# N_test * N_train
        begin = now()
        a = now()
        mask_time = 0
        sum_time= 0
        

        for i, x_leaf in enumerate(X_leaves): # n_test
            mask = (self.y_train_leaves_ != np.expand_dims(x_leaf, 1))
            x_weights = ma.masked_array(self.y_weights_, mask)# n_tree * n_train. for each n_test
            b = now()
            mask_time += b - a
            weights[i,:] = x_weights.sum(axis = 0)
            a = now()
            sum_time += a - b       
        # print("prediction iteration:", now()- begin, " with mask:", mask_time, "sum:", sum_time)
        if uv is 0: # debug. E(X_t|X_t-1). for CV too.
            return weights.dot(self.y_train_) / np.sum(weights,axis=1)[:,None]
        else:
            char_est_cos = weights.dot(np.cos(self.y_train_.dot(uv.T))) / np.sum(weights,axis=1)[:,None]
            char_est_sin = weights.dot(np.sin(self.y_train_.dot(uv.T))) / np.sum(weights,axis=1)[:,None]
        return char_est_cos, char_est_sin
 
class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """
    Based on BaseForestQuantileRegressor. What is the purpose?
    
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

class BaseTreeQuantileRegressor(BaseDecisionTree):
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """
        Child of BaseDecisionTree (sklearn), which use a single DecisionTree to do the same kind of Quantile things.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.


        Returns
        -------
        self : object
            Returns self.
        """
        # y passed from a forest is 2-D. This is to silence the
        # annoying data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=1)
        super(BaseTreeQuantileRegressor, self).fit(
            X, y, sample_weight=sample_weight, check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        self.y_train_ = y

        # Stores the leaf nodes that the samples lie in.
        self.y_train_leaves_ = self.tree_.apply(X)
        return self
    
    def predict(self, X,u, check_input=False): # ,cos_sin
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile is None:
            return super(BaseTreeQuantileRegressor, self).predict(X, check_input=check_input)

        B = u.shape[0]
        r_cos, r_sin = np.zeros((X.shape[0],B)), np.zeros((X.shape[0],B))
        X_leaves = self.apply(X)
        unique_leaves = np.unique(X_leaves)

        for leaf in unique_leaves:
            # for those X_test in that leaf, we use training in that leaf to cal the quantiles.
            y = self.y_train_[self.y_train_leaves_ == leaf]
            r_cos[X_leaves == leaf,:] = np.mean(np.cos(y.dot(uv.T)),axis = 0)
            r_sin[X_leaves == leaf,:] = np.mean(np.sin(y.dot(uv.T)),axis = 0)
        return r_cos, r_sin

class DecisionTreeQuantileRegressor(DecisionTreeRegressor, BaseTreeQuantileRegressor):
    """
    Just combine QBT and DecisionTreeRegressor, and provide _init_
    
    A decision tree regressor that provides quantile estimates.
    """
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None):
        super(DecisionTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)
