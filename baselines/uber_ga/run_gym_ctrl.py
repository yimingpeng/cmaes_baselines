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
    popsize = 33
    gensize = 2000
    truncation_size = 5
    sigma = 0.01
    eval_iters = 3
    from baselines.uber_ga import mlp_policy, ga_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    base_env = make_gym_control_env(env_id, seed)
    test_env = make_gym_control_env(env_id, seed)
    ga_simple.learn(base_env,
                       test_env,
                       policy_fn,
                       max_fitness = max_fitness,  # has to be negative, as cmaes consider minization
                       popsize = popsize,
                       gensize = gensize,
                       truncation_size = truncation_size,
                       sigma = sigma,
                       eval_iters = eval_iters,
                       max_timesteps=num_timesteps,
                       timesteps_per_actorbatch=2048,
                       seed=seed)
    base_env.close()
    test_env.close()


def main():
    args = gym_ctrl_arg_parser().parse_args()
    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "CMAES-"+args.env)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
