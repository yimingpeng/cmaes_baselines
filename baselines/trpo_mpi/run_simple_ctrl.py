#!/usr/bin/env python3
import inspect
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
# very important, don't remove, otherwise pybullet cannot run (reasons are unknown)
import pybullet_envs
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser, pybullet_arg_parser, make_pybullet_env, \
    gym_ctrl_arg_parser, make_gym_control_env
from baselines import logger
from baselines.ppo.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    # if rank == 0:
    #     logger.configure()
    # else:
    #     logger.configure(format_strs=[])
    #     logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_gym_control_env(env_id, seed)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    args = gym_ctrl_arg_parser().parse_args()
    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "TRPO-"+args.env+"_seed_"+str(args.seed))
    logger.log("Algorithm: TRPO-" + args.env + "_seed_" + str(args.seed))
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

