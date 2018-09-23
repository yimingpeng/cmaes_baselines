"""
Helpers for scripts like run_atari.py.
"""
import os, inspect
import pybullet

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import os
import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_gym_control_multi_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Added by Yiming (29/5/2018)
    Create a wrapped, monitored gym.Env for Simple Control Problems.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def make_gym_control_env(env_id, seed):
    """
    Added by Yiming (29/5/2018)
    Create a wrapped, monitored gym.Env for Simple Control Problems.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir(), allow_early_resets=True)
    env.seed(seed)
    return env

def make_pybullet_env(env_id, seed):
    """
    Added by Yiming (29/5/2018)
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    # pybullet.connect(None)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir(),allow_early_resets=True)
    # env = InvertedDoublePendulumBulletEnv()
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser

def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser

def gym_ctrl_arg_parser():
    """
    Added by Yiming (29/5/2018)
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    # parser.add_argument('--env', help='environment ID', type=str,
    #                     default="LunarLander-v2")
    # parser.add_argument('--env', help='environment ID', type=str,
        #                     default="LunarLanderContinuous-v2")
    parser.add_argument('--env', help='environment ID', type=str,
                        default="BipedalWalker-v2")
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    return parser


def pybullet_arg_parser():
    """
    Added by Yiming (29/5/2018)
    Create an argparse.ArgumentParser for run_pybullet.py.
    """
    parser = arg_parser()
    # parser.add_argument('--env', help='environment ID', type=str,
    #                     default="AntBulletEnv-v0")
    parser.add_argument('--env', help='environment ID', type=str,
                        default="InvertedPendulumSwingupBulletEnv-v0")
    # parser.add_argument('--env', help='environment ID', type=str,
    #                     default="HumanoidBulletEnv-v0")
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    return parser