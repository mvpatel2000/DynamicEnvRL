# DynamicEnvRL

This repo is currently in progress as research continues on "Optimizing Reinforcement Learning through Dynamic Environment Manipulation"

## notebooks
This folder contains various jupyter notebooks used in early stages of research. This includes tinkering and development of algorithms, various testing on ideas, and handling a small mountain of installation issues with regards to MuJoCo and rllab. This includes notebooks for the proposed duel training, static training, and dynamic training with one copy of each per benchmark problem. There is also a notebook for demos called sf-demo and viz_results_notebook. Various portions of code are from previous research, primarily in regards to graphing and visualizing policies.

## initial_results
This folder is a copy of the original trained policies from the notebooks. These were used to ensure the algorithms were properly written and everything was connected properly. These are also used for demos. The secondary policy v generated from duel training is stored here.

## scripts
This folder contains various python scripts for testing. These were developed after prototyping was completed in the jupyter notebooks. They support threading, queuing multiple experiments, and parallel sampling so they can be run for extended periods on a compute cluster.

## results
This folder stores output and results from the scripts.
