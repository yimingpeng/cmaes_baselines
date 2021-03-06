"""
Use a genetic algorithm to learn CartPole-v0.

Run with:

    $ mpirun -n 8 python3 -u run_simple_ctrl.py

You can change `8` to any value. It effects speed, but not
sample efficiency.
"""
#!/usr/bin/env python3
import inspect
import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# very important, don't remove, otherwise pybullet cannot run (reasons are unknown)
import pybullet_envs
import gym
import tensorflow as tf

from baselines import logger
from baselines.uber_ga_original import LearningSession, simple_mlp, make_session
from baselines.common.cmd_util import  make_pybullet_env, pybullet_arg_parser

POPULATION = 32

def main():
    """
    Train on CartPole.
    """

    args = pybullet_arg_parser().parse_args()

    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "Uber-GA-"+args.env+"_seed_"+str(args.seed))
    logger.log("Algorithm:Uber-GA-" + args.env + "_seed_" + str(args.seed))
    env_id = args.env
    seed = args.seed
    generation = 0
    with make_session() as sess:
        env = make_pybullet_env(env_id, seed)
        try:
            model = simple_mlp(sess, env)
            sess.run(tf.global_variables_initializer())
            learn_sess = LearningSession(sess, model)
            while True:
                if generation >= 10000 or learn_sess.timesteps_so_far >= 5e6:
                    break
                pop = learn_sess.generation(env, trials=1, population=POPULATION)
                generation +=1
                # rewards = [x[0] for x in pop]
                # print('mean=%f best=%s' % (sum(rewards)/len(rewards), str(rewards[:10])))
        finally:
            env.close()

if __name__ == '__main__':
    main()
