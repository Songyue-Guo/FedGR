'''
@ Author: Yue 
@ Date: 2022-06-21 08:17:03
@ LastEditTime: 2022-06-22 08:10:19
@ FilePath: /gsy/FedDUB/config.py
@ Description: 
@ Email: 21S151140@stu.hit.edu.cn , Tel: +86-13184023012 
@ Copyright (c) 2022 by HITSZ / Songyue Guo, All Rights Reserved. 
'''

default_param_dicts = {
    "famnist": {
        "dataset": ["famnist"],
        "split": ["dirichlet"],
        "dir_alpha": [None],
        "n_classes": [10],
        "n_clients": [None],
        "n_max_sam": [None],
        "fed_ratio": [None],
        "net": [None],
        "max_round": [None],
        "test_round": [None],
        "local_epochs": [None],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cifar10": {
        "dataset": ["cifar10"],
        "split": ["dirichlet"],
        "dir_alpha": [None],
        "n_classes": [10],
        "n_clients": [None],
        "n_max_sam": [None],
        "fed_ratio": [None],
        "net": [None],
        "max_round": [None],
        "test_round": [None],
        "local_epochs": [None],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
    "cifar100": {
        "dataset": ["cifar100"],
        "split": ["dirichlet"],
        "dir_alpha": [None],
        "n_classes": [100],
        "n_clients": [None],
        "n_max_sam": [None],
        "c_ratio": [None],
        "net": [None],
        "max_round": [None],
        "test_round": [None],
        "local_epochs": [None],
        "local_steps": [None],
        "batch_size": [64],
        "optimizer": ["SGD"],
        "lr": [0.03],
        "momentum": [0.9],
        "weight_decay": [1e-5],
        "max_grad_norm": [100.0],
        "cuda": [True],
    },
}
