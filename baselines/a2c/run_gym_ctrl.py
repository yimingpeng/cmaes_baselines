#!/usr/bin/env python3
# Add the current folder to PYTHONPATH by Yiming
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(),os.pardir)), os.pardir)))

from baselines.common.cmd_util import gym_ctrl_arg_parser, make_gym_control_multi_env
from baselines import logger
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy,MlpPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, env_name):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'mlp':
        policy_fn = MlpPolicy
    env = make_gym_control_multi_env(env_id, num_env, seed)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, env_name=env_name)
    env.close()


def main():

    parser = gym_ctrl_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='mlp')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear')
    args = parser.parse_args()
    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "A2C-"+args.env)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=1, env_name=args.env)

if __name__ == '__main__':
    main()
