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
"""
Genetic algorithm outer loop.
"""

# Avoid MPI errors:
# pylint: disable=E1101

from mpi4py import MPI
import tensorflow as tf
import numpy as np
import time
from baselines import logger
from .noise import NoiseSource, NoiseAdder, noise_seed
from .selection import truncation_selection
from collections import deque

class LearningSession:
    """
    A GA optimization session.
    """
    def __init__(self, session, model, noise=None, selection=truncation_selection):
        self.session = session
        self.model = model
        self.population = None
        self.selection = selection
        self._noise_adder = NoiseAdder(self.session, self.model.variables, noise or NoiseSource())

        self.timesteps_so_far = 0
        self.episodes_so_far = 0
        self.tstart =time.time()
        self.rewbuffer = deque(maxlen = 100)
        self.lenbuffer = deque(maxlen = 100)
        _synchronize_variables(self.session, self.model.variables)

    def export_state(self):
        """
        Export the state of the learning session to a
        picklable object.
        """
        return {
            'model': self.model.export_state(),
            'population': self.population,
            'noise': self._noise_adder.noise
        }

    def import_state(self, state):
        """
        Import a state exported by export_state().

        This assumes that the LearningSession has already
        been setup with a suitable TensorFlow graph.

        This may add nodes (e.g. assigns) to the graph, so
        it should not be called often.
        """
        self.model.import_state(state['model'])
        self.population = state['population']
        self._noise_adder.noise = state['noise']

    # pylint: disable=R0913
    def generation(self, env, trials=1, population=5000, stddev=0.1, **select_kwargs):
        """
        Run a generation of the algorithm and update the
        population accordingly.

        Call this from each MPI worker.

        Args:
          env: the gym.Env to use to evaluate the model.
          trials: the number of episodes to run.
          population: the number of new genomes to try.
          stddev: mutation standard deviation.
          select_kwargs: kwargs for selection algorithm.

        Updates self.population to a sorted list of
        (fitness, genome) tuples.

        Returns the new population for backwards
        compatibility.
        """
        eval_seq = self.traj_segment_generator_eval(env, 2048)
        selected = self._select(population, select_kwargs)
        res = []
        for i in range(MPI.COMM_WORLD.Get_rank(), population+1, MPI.COMM_WORLD.Get_size()):
            if i == 0 and self.population is not None:
                mutations = self.population[0][1]
            else:
                mutations = selected[i - 1] + ((noise_seed(), stddev),)
            res.append((self.evaluate(mutations, env, trials, step_fn=None, eval_seq=eval_seq), mutations))
        full_res = [x for batch in MPI.COMM_WORLD.allgather(res) for x in batch]
        self.population = sorted(full_res, reverse=True)
        self.generation+=1
        return self.population

    def traj_segment_generator_eval(self, env, horizon):
        import numpy as np
        t = 0
        ac = env.action_space.sample() # not used, just so we have the datatype
        new = True # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_len = 0 # len of current episode
        ep_rets = [] # returns of completed episodes in this segment
        ep_lens = [] # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()
        state = self.model.start_state(1)

        while True:
            prevac = ac
            out = self.model.step([obs], state)
            state = out['states']
            ac = out['actions'][0]
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "new" : news,
                        "ac" : acs, "prevac" : prevacs,
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            obs, rew, done, _ = env.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                state = self.model.start_state(1)
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1

    def evaluate(self, mutations, env, trials, step_fn=None, eval_seq=None):
        """
        Evaluate a genome on an environment.

        Args:
          mutations: a list of (seed, stddev) tuples.
          env: the environment to run.
          trials: the number of episodes to run.
          step_fn: a function to call before each step.

        Returns:
          The mean reward over all the trials.
        """
        with self._noise_adder.seed(mutations):
            self.model.variables_changed()
            rewards = []
            for _ in range(trials):
                done = False
                total_rew = 0.0
                state = self.model.start_state(1)
                obs = env.reset()
                record = False
                while not done:
                    if self.timesteps_so_far % 10000 == 0:
                        record = True
                    if step_fn:
                        step_fn()
                    out = self.model.step([obs], state)
                    state = out['states']
                    obs, rew, done, _ = env.step(out['actions'][0])
                    total_rew += rew
                    self.timesteps_so_far += 1
                if record:
                    self.rewbuffer.extend(rewards)
                    self.lenbuffer.extend(rewards)
                    self.result_record()
                    record = False
                rewards.append(total_rew)
                self.episodes_so_far += 1
            # print(self.timesteps_so_far)
            return sum(rewards) / len(rewards)

    def _select(self, children, select_kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            if self.population is None:
                selected = [()] * children
            else:
                selected = self.selection(self.population, children, **select_kwargs)
            MPI.COMM_WORLD.bcast(selected)
            return selected
        return MPI.COMM_WORLD.bcast(None)

    def result_record(self):
        # if best_fitness != -np.inf:
        #     rewbuffer.append(best_fitness)
        print(self.rewbuffer)
        if len(self.lenbuffer) == 0:
            mean_lenbuffer = 0
        else:
            mean_lenbuffer = np.mean(self.lenbuffer)
        if len(self.rewbuffer) == 0:
            # TODO: Add pong game checking
            mean_rewbuffer = 0
        else:
            mean_rewbuffer = np.mean(self.rewbuffer)
        logger.record_tabular("EpLenMean", mean_lenbuffer)
        logger.record_tabular("EpRewMean", mean_rewbuffer)
        logger.record_tabular("EpisodesSoFar", self.episodes_so_far)
        logger.record_tabular("TimestepsSoFar", self.timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - self.tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

def _synchronize_variables(sess, variables):
    if MPI.COMM_WORLD.Get_rank() == 0:
        for var in variables:
            MPI.COMM_WORLD.bcast(sess.run(var))
    else:
        for var in variables:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))
