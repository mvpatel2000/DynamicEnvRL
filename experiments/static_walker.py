from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv

from multiprocessing import Process, Queue
import numpy as np
import os
import gym
import random
import pickle
import scipy
import argparse

def train(num_experiments, thread_id, queue):

    ############ DEFAULT PARAMETERS ############

    env_name = None                     #Name of adversarial environment
    path_length = 1000                  #Maximum episode length
    layer_size = tuple([100,100,100])   #Layer definition
    ifRender = False                    #Should we render?
    afterRender = 100                   #After how many to animate
    n_exps = 1                          #Number of training instances to run
    n_itr = 25                          #Number of iterations of the alternating optimization
    n_pro_itr = 1                       #Number of iterations for the protaginist
    n_adv_itr = 1                       #Number of interations for the adversary
    batch_size = 4000                   #Number of training samples for each iteration
    ifSave = True                       #Should we save?
    save_every = 100                    #Save checkpoint every save_every iterations
    n_process = 1                       #Number of parallel threads for sampling environment
    adv_fraction = 0.25                 #Fraction of maximum adversarial force to be applied
    step_size = 0.01                    #kl step size for TRPO
    gae_lambda = 0.97                   #gae_lambda for learner
    save_dir = './results'              #folder to save result in

    ############ ENV SPECIFIC PARAMETERS ############

    env_name = 'Walker2dAdv-v1'

    layer_size = tuple([64,64])
    step_size = 0.1
    gae_lambda = 0.97
    batch_size = 25000

    n_exps = num_experiments
    n_itr = 500
    ifSave = False
    n_process = 4

    adv_fraction = 5.0

    save_dir = './../results/StaticWalker'

    args = [env_name, path_length, layer_size, ifRender, afterRender, n_exps, n_itr, n_pro_itr, n_adv_itr, 
            batch_size, save_every, n_process, adv_fraction, step_size, gae_lambda, save_dir]

    ############ ADVERSARIAL POLICY LOAD ############

    filepath = './../initial_results/Walker/env-Walker2dAdv-v1_Exp1_Itr1500_BS25000_Adv0.25_stp0.01_lam0.97_507500.p'
    res_D = pickle.load(open(filepath,'rb'))
    pretrained_adv_policy = res_D['adv_policy']

    ############ MAIN LOOP ############

    ## Initializing summaries for the tests ##
    const_test_rew_summary = []
    rand_test_rew_summary = []
    step_test_rew_summary = []
    rand_step_test_rew_summary = []
    adv_test_rew_summary = []

    ## Preparing file to save results in ##
    save_prefix = 'static_env-{}_Exp{}_Itr{}_BS{}_Adv{}_stp{}_lam{}_{}'.format(env_name, n_exps, n_itr, batch_size, adv_fraction, step_size, gae_lambda, random.randint(0,1000000))
    save_name = save_dir+'/'+save_prefix

    ## Looping over experiments to carry out ##
    for ne in range(n_exps):
        ## Environment definition ##
        ## The second argument in GymEnv defines the relative magnitude of adversary. For testing we set this to 1.0.
        env = normalize(GymEnv(env_name, adv_fraction))
        env_orig = normalize(GymEnv(env_name, 1.0))

        ## Protagonist policy definition ##
        pro_policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=layer_size,
            is_protagonist=True
        )
        pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

        ## Zero Adversary for the protagonist training ##
        zero_adv_policy = ConstantControlPolicy(
            env_spec=env.spec,
            is_protagonist=False,
            constant_val = 0.0
        )

        ## Adversary policy definition ##
        adv_policy = pretrained_adv_policy
        adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

        ## Initializing the parallel sampler ##
        parallel_sampler.initialize(n_process)

        ## Optimizer for the Protagonist ##
        pro_algo = TRPO(
            env=env,
            pro_policy=pro_policy,
            adv_policy=adv_policy,
            pro_baseline=pro_baseline,
            adv_baseline=adv_baseline,
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_pro_itr,
            discount=0.995,
            gae_lambda=gae_lambda,
            step_size=step_size,
            is_protagonist=True
        )

        ## Setting up summaries for testing for a specific training instance ##
        pro_rews = []
        adv_rews = []
        all_rews = []
        const_testing_rews = []
        const_testing_rews.append(test_const_adv(env_orig, pro_policy, path_length=path_length))
        rand_testing_rews = []
        rand_testing_rews.append(test_rand_adv(env_orig, pro_policy, path_length=path_length))
        step_testing_rews = []
        step_testing_rews.append(test_step_adv(env_orig, pro_policy, path_length=path_length))
        rand_step_testing_rews = []
        rand_step_testing_rews.append(test_rand_step_adv(env_orig, pro_policy, path_length=path_length))
        adv_testing_rews = []
        adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))

        ## Beginning alternating optimization ##
        for ni in range(n_itr):
            logger.log('\n\nThread: {} Experiment: {} Iteration: {}\n'.format(thread_id, ne,ni,))
            
            ## Train Protagonist
            pro_algo.train()
            pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
            logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
            
            ## Test the learnt policies
            const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
            rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
            step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
            rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
            adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))

            if ni%afterRender==0 and ifRender==True:
                test_const_adv(env, pro_policy, path_length=path_length, n_traj=1, render=True);

            if ni!=0 and ni%save_every==0 and ifSave==True:
                ## SAVING CHECKPOINT INFO ##
                pickle.dump({'args': args,
                             'pro_policy': pro_policy,
                             'adv_policy': adv_policy,
                             'zero_test': [const_testing_rews],
                             'rand_test': [rand_testing_rews],
                             'step_test': [step_testing_rews],
                             'rand_step_test': [rand_step_testing_rews],
                             'iter_save': ni,
                             'exp_save': ne,
                             'adv_test': [adv_testing_rews]}, open(save_name+'_'+str(ni)+'.p','wb'))

        ## Shutting down the optimizer ##
        pro_algo.shutdown_worker()

        ## Updating the test summaries over all training instances
        const_test_rew_summary.append(const_testing_rews)
        rand_test_rew_summary.append(rand_testing_rews)
        step_test_rew_summary.append(step_testing_rews)
        rand_step_test_rew_summary.append(rand_step_testing_rews)
        adv_test_rew_summary.append(adv_testing_rews)

    queue.put([const_test_rew_summary, rand_test_rew_summary, step_test_rew_summary, rand_step_test_rew_summary, adv_test_rew_summary])


    ############ SAVING MODEL ############
    '''
    with open(save_name+'.p','wb') as f:
        pickle.dump({'args': args,
            'pro_policy': pro_policy,
            'adv_policy': adv_policy,
            'zero_test': const_test_rew_summary,
            'rand_test': rand_test_rew_summary,
            'step_test': step_test_rew_summary,
            'rand_step_test': rand_step_test_rew_summary,
            'adv_test': adv_test_rew_summary}, f)
    '''

if __name__ == '__main__':
    
    num_threads = 10
    n_exps = 5

    queue = Queue()
    pool = []
    results = []

    for i in range(num_threads):
        p = Process(target=train, args=(n_exps, i, queue,))
        pool.append(p)
        
    for process in pool:
        process.start()

    for process in pool:
        results.append(queue.get())

    for process in pool:
        process.join()

    summary = [[] for x in range(5)]
    for result in results:
        for i, exps in enumerate(result):
            for exp in exps:
                summary[i].append(exp)


    with open('./../results/StaticWalker/static_walker'+'.p','wb') as f:
        pickle.dump({
            'zero_test': summary[0],         #const_test_rew_summary
            'rand_test': summary[1],         #rand_test_rew_summary
            'step_test': summary[2],         #step_test_rew_summary
            'rand_step_test': summary[3],    #rand_step_test_rew_summary
            'adv_test': summary[4],          #adv_test_rew_summary
            }, f)
