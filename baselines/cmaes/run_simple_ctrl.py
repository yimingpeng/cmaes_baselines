#!/usr/bin/env python3
# Add the current folder to PYTHONPATH by Yiming
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))

from baselines.common.cmd_util import simple_ctrl_arg_parser, \
    make_simple_control_env
from baselines.common import tf_util as U
from baselines import logger


def train(env_id, num_timesteps, seed):
    max_fitness = -10000
    popsize = 32
    gensize = 200
    bounds = [-5.0, 5.0]
    sigma = 0.1
    eval_iters = 3
    from baselines.cmaes import mlp_policy, cmaes_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space,
                                    hid_size=16, num_hid_layers=2)

    base_env = make_simple_control_env(env_id, seed)
    cmaes_simple.learn(base_env,
                       policy_fn,
                       max_fitness = max_fitness,  # has to be negative, as cmaes consider minization
                       popsize = popsize,
                       gensize = gensize,
                       bounds = bounds,
                       sigma = sigma,
                       eval_iters = eval_iters,
                       max_timesteps=num_timesteps,
                       timesteps_per_actorbatch=2048,
                       seed=seed)
    base_env.close()


def main():
    args = simple_ctrl_arg_parser().parse_args()
    logger.configure(filename="PPO1-" + args.env,
                     format_strs=['stdout', 'log', 'csv'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
