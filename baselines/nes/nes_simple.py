import time
from collections import deque

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

import baselines.common.tf_util as U
import cma
import numpy as np
from baselines import logger
from mpi4py import MPI
import tensorflow as tf
from baselines.common import zipsame, set_global_seeds


def traj_segment_generator_eval(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "new": news,
                   "ac": acs, "prevac": prevacs,
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        if env.spec._env_name == "LunarLanderContinuous":
            ac = np.clip(ac, -1.0, 1.0)
        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def traj_segment_generator(pi, env, horizon, stochastic, eval_iters):
    global timesteps_so_far
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    ep_num = 0
    while True:
        if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
            result_record()
        prevac = ac
        ac = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if (t > 0 and t % horizon == 0) or ep_num >= eval_iters:
            yield {"ob": obs, "rew": rews, "new": news,
                   "ac": acs, "prevac": prevacs,
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            if ep_num >= eval_iters:
                ep_num = 0
                t = 0
        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        if env.spec._env_name == "LunarLanderContinuous":
            ac = np.clip(ac, -1.0, 1.0)
        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        timesteps_so_far += 1
        if new:
            ep_num += 1
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def result_record():
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart, best_fitness
    if best_fitness != -np.inf:
        rewbuffer.append(best_fitness)
    if len(lenbuffer) == 0:
        mean_lenbuffer = 0
    else:
        mean_lenbuffer = np.mean(lenbuffer)
    if len(rewbuffer) == 0:
        # TODO: Add pong game checking
        mean_rewbuffer = 0
    else:
        mean_rewbuffer = np.mean(rewbuffer)
    logger.record_tabular("EpLenMean", mean_lenbuffer)
    logger.record_tabular("EpRewMean", mean_rewbuffer)
    logger.record_tabular("EpisodesSoFar", episodes_so_far)
    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
    logger.record_tabular("TimeElapsed", time.time() - tstart)
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.dump_tabular()


def learn(base_env,
          policy_fn, *,
          max_fitness,  # has to be negative, as cmaes consider minization
          popsize,
          gensize,
          bounds,
          sigma,
          eval_iters,
          timesteps_per_actorbatch,
          max_timesteps = 0,
          max_episodes = 0,
          max_iters = 0,
          max_seconds = 0,
          seed = 0,
          optim_stepsize = 3e-4,
          schedule='constant' # annealing for stepsize parameters (epsilon and adam)
          ):
    set_global_seeds(seed)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = base_env.observation_space
    ac_space = base_env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    backup_pi = policy_fn("backup_pi", ob_space,
                          ac_space)  # Construct a network for every individual to adapt during the es evolution

    sol_dim = int(np.sum([np.prod(v.get_shape().as_list()) for v in pi.get_trainable_variables()]))
    pop_size = tf.placeholder(dtype = tf.float32, shape = [])
    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    tfkids_fit = tf.placeholder(dtype = tf.float32, shape = [popsize,])
    tfkids = tf.placeholder(dtype = tf.float32, shape = [popsize, sol_dim])

    tfmean = tf.Variable(initial_value = tf.random_normal([sol_dim, ], 0., 1.), dtype=tf.float32)
    tfcov = tf.Variable(initial_value = tf.eye(sol_dim), dtype = tf.float32)
    mvn = MultivariateNormalFullCovariance(loc = tfmean, covariance_matrix = tfcov)

    loss = -tf.reduce_mean(mvn.log_prob(tfkids) * tfkids_fit)
    train_op = tf.train.GradientDescentOptimizer(lrmult).minimize(loss)

    optimize = U.function([tfkids, tfkids_fit, lrmult], [train_op])
    reproduce = U.function([pop_size], [mvn.sample(popsize)])
    get_mean = U.function([], [tfmean])

    input_mean = tf.placeholder(dtype = tf.float32, shape = [sol_dim, ])
    assign_weights_to_mean = U.function([input_mean], [tf.assign(tfmean, input_mean)])

    U.initialize()

    pi_set_from_flat_params = U.SetFromFlat(pi.get_trainable_variables())
    pi_get_flat_params = U.GetFlat(pi.get_trainable_variables())

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer, best_fitness, eval_seq
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards

    assign_backup_eq_new = U.function([], [], updates = [tf.assign(backup_v, newv)
                                                         for (backup_v, newv) in zipsame(
            backup_pi.get_variables(), pi.get_variables())])
    assign_new_eq_backup = U.function([], [], updates = [tf.assign(newv, backup_v)
                                                         for (newv, backup_v) in zipsame(
            pi.get_variables(), backup_pi.get_variables())])

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    # Build generator for all solutions
    actors = []
    best_fitness = -np.inf

    eval_seq = traj_segment_generator_eval(pi, base_env,
                                           timesteps_per_actorbatch,
                                           stochastic = True)
    for i in range(popsize):
        newActor = traj_segment_generator(pi, base_env,
                                          timesteps_per_actorbatch,
                                          stochastic = True,
                                          eval_iters = eval_iters)
        actors.append(newActor)
    while True:
        if max_timesteps and timesteps_so_far >= max_timesteps:
            logger.log("Max time steps")
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            logger.log("Max episodes")
            break
        elif max_iters and iters_so_far >= max_iters:
            logger.log("Max iterations")
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            logger.log("Max time")
            break
        assign_backup_eq_new()  # backup current policy

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / (max_timesteps/2), 0)
        else:
            raise NotImplementedError

        logger.log("********** Generation %i ************" % iters_so_far)
        eval_seg = eval_seq.__next__()
        rewbuffer.extend(eval_seg["ep_rets"])
        lenbuffer.extend(eval_seg["ep_lens"])
        if iters_so_far == 0:
            result_record()
            assign_weights_to_mean(pi_get_flat_params())
        # mean = pi_get_flat_params()
        solutions = reproduce(popsize)
        ob_segs = None
        segs = []
        costs = []
        lens = []
        for id, solution in enumerate(solutions[0]):
            # pi.set_Flat_variables(solution)
            pi_set_from_flat_params(solution)
            seg = actors[id].__next__()
            costs.append(-np.mean(seg["ep_rets"]))
            lens.append(np.sum(seg["ep_lens"]))
            segs.append(seg)
            if ob_segs is None:
                ob_segs = {'ob': np.copy(seg['ob'])}
            else:
                ob_segs['ob'] = np.append(ob_segs['ob'], seg['ob'], axis = 0)
            assign_new_eq_backup()
        optimize(solutions[0], np.array(costs), cur_lrmult * optim_stepsize)
        # fit_idx = np.array(costs).flatten().argsort()[:len(costs)]
        # solutions = np.array(solutions)[fit_idx]
        # costs = np.array(costs)[fit_idx]
        # segs = np.array(segs)[fit_idx]
        # # Weights decay
        # # costs, real_costs = fitness_shift(costs)
        # # costs, real_costs = compute_centered_ranks(costs)
        # l2_decay = compute_weight_decay(0.01, solutions)
        # costs += l2_decay
        # costs, real_costs = fitness_normalization(costs)
        # # best_solution = np.copy(solutions[0])
        # # best_fitness = -real_costs[0]
        # # rewbuffer.extend(segs[0]["ep_rets"])
        # # lenbuffer.extend(segs[0]["ep_lens"])
        # es.tell_real_seg(solutions = solutions, function_values = costs, real_f = real_costs, segs = segs)
        # best_solution = np.copy(es.result[0])
        # best_fitness = -es.result[1]
        # rewbuffer.extend(es.result[3]["ep_rets"])
        # lenbuffer.extend(es.result[3]["ep_lens"])
        # logger.log("Generation:", es.countiter)
        # logger.log("Best Solution Fitness:", best_fitness)
        pi_set_from_flat_params(get_mean()[0])

        ob = ob_segs["ob"]
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for observation normalization

        iters_so_far += 1
        episodes_so_far += sum(lens)


def fitness_shift(x):
    x = np.asarray(x).flatten()
    ranks = np.empty(len(x))
    ranks[x.argsort()] = np.arange(len(x))
    ranks /= (len(x) - 1)
    ranks -= .5
    return ranks, x


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis = 1)


def fitness_normalization(x):
    x = np.asarray(x).flatten()
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, x


def compute_centered_ranks(x):
    """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y, x


def compute_ranks(x):
    """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype = int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
