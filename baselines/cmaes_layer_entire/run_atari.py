#!/usr/bin/env python3
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser


# logger.reset()
# logger.configure(filename="PPO1-Breakout",
#                  format_strs=['stdout', 'log', 'csv'])
def train(env_id, num_timesteps, seed):
    from baselines.cmaes_simple import cmaes_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = make_atari(env_id)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))
    # test_env = bench.Monitor(test_env, logger.get_dir() and
    #     osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    cmaes_simple.learn(env, policy_fn,
                       max_timesteps=int(num_timesteps * 1.1),
                       timesteps_per_actorbatch=256,
                       clip_param=0.1, entcoeff=0.01,
                       optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                       gamma=0.99, lam=0.95,
                       schedule='linear', seed=seed,
                       env_id=env_id
                       )
    env.close()


def main():
    args = atari_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
