
# Does the Markov Decision Process Fit the Data: Testing for the Markov Property in Sequential Decision Making

This repository contains the implementation for the paper "Does the Markov Decision Process Fit the Data: Testing for the Markov Property in Sequential Decision Making" (ICML 2020) in Python.

## Summary of the paper

The Markov assumption (MA) is fundamental to the empirical validity of reinforcement learning. In this paper, we propose a novel Forward-Backward Learning procedure to test MA in sequential decision making. The proposed test does not assume any parametric form on the joint distribution of the observed data and plays an important role for identifying the optimal policy in high-order Markov decision processes and partially observable MDPs. We apply our test to both synthetic datasets and a real data example from mobile health studies to illustrate its usefulness.

<img align="center" src="diag.png" alt="drawing" width="600">



## Requirements
Change your working directory to this main folder, run `conda env create --file TestMDP.yml` to create the Conda environment, and then run `conda activate TestMDP` to activate the environment.

## File Overview
2. `/test_func`: main functions for the proposed test
    1. `_core_test_fun.py`: main functions for the proposed test, including Algorithm 1 and 2 in the paper, and their componnets.
    5. `_QRF.py`: the random forests regressor used in our experiments.
    6. `_uti_basic.py` and `_utility.py`: helper functions
1. `/experiment_script`: scripts for reproducing results. See next section. 
2. `/experiment_func`: supporting functions for the experiments presented in the paper
        2. `_DGP_Ohio.py`: simulate data and evaluate policies for the HMDP synthetic data section.
        3. `_DGP_TIGER.py`: simulate data for the POMDP synthetic data section.
        7. `_utility_RL.py`: RL algorithms used in the experiments, including FQI, FQE and related functions.

## How to reproduce results in the paper
Simply run the corresponding scripts:

1. Figure 2: `Ohio_simu_testing.py`
2. Figure 3: `Ohio_simu_values.py` and `Ohio_simu_seq_lags.py`
3. Figure 4: `Tiger_simu.py`

## How to test the Markov property for your own data
1. run `from _core_test_fun import *` to import required functions
2. Algorithm 1: decide whether or not your data satisfies J-th order Markov property
    1. make sure your data, the observed trajectories, is a list of [X, A], each for one trajectory. Here, X is a T by dim_state_variable array for observed states, and A is a T by dim_action_variable array for observed actions. 
    2. run `test(data = data, J = J)`, and the output is the p-value. More optional parameters can be found in the file. 
3. Algorithm 2: decide whether the system is an MDP (and its order) or the system is most likely to be a POMDP 
    1. make sure your data and parameters satisfy the requirement for  `test()`. 
    2. specify the significance level alpha and order upper bound K. 
    2. run `selectOrder(data = data, K = K, alpha = alpha)`. More optional parameters can be found in the file. 



## Citation

Please cite our paper
[Does the Markov Decision Process Fit the Data: Testing for the Markov Property in Sequential Decision Making (ICML 2020)](https://arxiv.org/abs/2002.01751)

``` 
@article{Shi2020DoesTM,
  title={Does the Markov Decision Process Fit the Data: Testing for the Markov Property in Sequential Decision Making},
  author={Chengchun Shi and Runzhe Wan and Rui Song and Wenbin Lu and Ling Leng},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.01751}
}
``` 


## Contributing

All contributions welcome! All content in this repository is licensed under the MIT license.

