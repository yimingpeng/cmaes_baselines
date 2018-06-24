#!/usr/bin/env python3
# Add the current folder to PYTHONPATH by Yiming
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))

from baselines.common.cmd_util import gym_ctrl_arg_parser, \
    make_gym_control_env
from baselines.common import tf_util as U
from baselines import logger


def train(env_id, num_timesteps, seed):
    max_fitness = -100000
    popsize = 32
    gensize = 20 # gen size for each iteration
    bounds = [-5.0, 5.0]
    max_v_train_iter = 10
    sigma = 0.01
    eval_iters = 1
    from baselines.ppo_cmaes_surrogate2_V import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = make_gym_control_env(env_id, seed)
    test_env = make_gym_control_env(env_id, seed)
    pposgd_simple.learn(env, test_env, policy_fn,
                        max_fitness = max_fitness,  # has to be negative, as cmaes consider minization
                        popsize = popsize,
                        gensize = gensize,
                        bounds = bounds,
                        sigma = sigma,
                        eval_iters = eval_iters,
                        max_v_train_iter = max_v_train_iter,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=10, optim_stepsize=3e-4,
                        optim_batchsize=64,
                        gamma=0.99, lam=0.95, schedule='linear', seed=seed,
                        env_id=env_id)
    env.close()
    test_env.close()


def main():
    args = gym_ctrl_arg_parser().parse_args()
    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "PPO_CMAES-"+args.env)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
