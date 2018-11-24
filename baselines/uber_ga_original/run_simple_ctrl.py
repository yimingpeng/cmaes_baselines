"""
Use a genetic algorithm to learn CartPole-v0.

Run with:

    $ mpirun -n 8 python3 -u run_simple_ctrl.py

You can change `8` to any value. It effects speed, but not
sample efficiency.
"""

import gym
import tensorflow as tf

import logger
from baselines.uber_ga_original import LearningSession, simple_mlp, make_session
from common.cmd_util import make_gym_control_env, gym_ctrl_arg_parser

POPULATION = 10

def main():
    """
    Train on CartPole.
    """

    args = gym_ctrl_arg_parser().parse_args()

    logger.configure(format_strs=['stdout', 'log', 'csv'], log_suffix = "UberGA-"+args.env+"_seed_"+str(args.seed))
    logger.log("Algorithm:UberGA-" + args.env + "_seed_" + str(args.seed))
    env_id = args.env
    seed = args.seed
    with make_session() as sess:
        env = make_gym_control_env(env_id, seed)
        try:
            model = simple_mlp(sess, env)
            sess.run(tf.global_variables_initializer())
            learn_sess = LearningSession(sess, model)
            while True:
                pop = learn_sess.generation(env, trials=5, population=POPULATION)
                rewards = [x[0] for x in pop]
                print('mean=%f best=%s' % (sum(rewards)/len(rewards), str(rewards[:10])))
        finally:
            env.close()

if __name__ == '__main__':
    main()