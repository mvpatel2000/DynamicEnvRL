# DynamicEnvRL
This document is an expanded version of the README.md from the submitted github repository zip. It fleshes out individual components in greater depth.<br/>

A brief overview: the rllab folder is a modified fork of the public repository rllab. This contains the code to sample from environments, update policies, implementation of TRPO, testing code and far more. The vast majority of software is within here so the primary scripts are relatively small and readable. The notebooks folder contains various jupyter notebooks for prototyping and visualizations. The experiments folder contains the jupyter notebook code converted to a single file with the inclusion of threads for parallelization. Initial Results has the duel policies and was used for prototyping. Results contains the final results. Other notebooks exist outside folders for result analysis.<br/>

Note that running the software requires several installations. The primary ones are OpenAI Gym, MuJoCo, and jupyter notebook. Adding rllab to the python path is also required. There are several additional dependencies that can easily be installed with apt install on a linux machine such as theano, lasagne, and tensorflow.

## rllab
The local folder of rllab is the one used in this research. This is a fork that blends together pieces from rllab-adv, an offshoot that was developed by lerrel but discontinued. It also contains several modernizations and upgrades as rllab has not been sufficiently maintained. Upon completion of this research, a pull request will be submitted that includes these upgrades. A new version of rllab-adv will also be released. <br/>

This framework is rather extensive and complicated. Slight updates are made to almost every subfolder to allow for handling two policies within one environment. The key point of modification is under rllab/rllab/policies. This includes some algorithmic variations and other testings that allow for two policies.

This version of rllab-adv depends on gym-adv, a fork of OpenAI Gym. This version of gym-adv allows for two policies within an environments and adds in opposing forces into each problem to test dynamic environment manipulation acceleration.

## notebooks
This folder contains various jupyter notebooks used in early stages of research. This includes tinkering and development of algorithms, various testing on ideas, and handling a small mountain of installation issues with regards to MuJoCo and rllab. This includes notebooks for the proposed duel training, static training, and dynamic training with one copy of each per benchmark problem. These all follow the algorithm descriptions in the paper.<br/>

There is also a notebook for demos called sf-demo and viz_results_notebook. These are used for visualizations of the learned policies and launch a window which renders the MuJoCo world and plays the policy attempting to move. They also do minimal graphing to confirm proper training. These were used to verify all steps were done properly and no logic bugs had been created when modifying rllab.
Various portions of code are from previous research, primarily in regards to graphing and visualizing policies.

## initial_results
This folder is a copy of the original trained policies from the notebooks. These were used to ensure the algorithms were properly written and everything was connected properly. These are also used for demos. <br/>

The secondary policy v generated from duel training is also stored here. As the generation of v does not need to be replicated multiple times, the original policy trained from the jupyter notebook was kept to save computation time. There are subfolders in for each benchmark problem to nicely separate pickle files. These pickle files include the final policy along with checkpoints every 20 iterations. As initial testing was run on a laptop, the model was reloaded from a checkpoint and continued whenever the laptop had to be closed for some period of time. 

## experiments
This folder contains various python scripts for testing. These were developed after prototyping was completed in the jupyter notebooks. They support threading, queuing multiple experiments, and parallel sampling so they can be run for extended periods on a compute cluster. This is to get multiple trials as RL is notoriously noisy.<br/>

The code in these scripts is nearly identical to the final versions of the jupyter notebooks. The entire training procedure is encompassed by a method called train which is passed as an argument for each process. The multiprocessing library is used for threading.
For parallelization, each script calls n threads which each run m experiments one after another. The values used for the current paper are 12 threads with 3 experiments. This parallelization ensured trials finished in a reasonable time and did not take several months. The parallelized code was run on TJHSST servers and dispersed over 32 cores. Typically, two scripts were run at a time.

## results
This folder stores the output and results from the scripts. This consists of a single pickle file which contains the aggregate results from all 36 trials for that benchmark problem. These are later loaded into other jupyter notebooks for analysis.

## Other Notebooks
There are four jupyter notebooks outside of any folder. These are used for generating graphs and analysis of results collected from the scripts. This includes reward per iteration graph, advantage in reward graph, percentile distribution of final policies graph, and images captured from visualizations. They are also used for various demonstration purposes at presentations.

# Future Work
Currently, more sample and demo codes are being developed to allow other users to verify and incorporate this technique in a more streamlined fashion. The research will be submitted to CoRL 2018.
