import time
from collections import deque

import baselines.common.tf_util as U
import cma
import numpy as np
from baselines import logger
from mpi4py import MPI

def traj_segment_generator_eval(pi, env, horizon, stochastic):
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

    while True:
        prevac = ac
        ac = pi.act(stochastic, ob)
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
    global timesteps_so_far, best_fitness
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

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        timesteps_so_far += 1
        if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
            result_record(best_fitness)
        if new:
            ep_num += 1
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def result_record(best_fitness_so_far):
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
    logger.log("********** Iteration %i ************" % iters_so_far)
    rewbuffer.append(best_fitness_so_far)
    print(rewbuffer)
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
          seed = 0
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = base_env.observation_space
    ac_space = base_env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy

    U.initialize()
    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer,best_fitness
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    # Build generator for all solutions
    actors = []
    best_fitness = 0
    for i in range(popsize):
        newActor = traj_segment_generator(pi, base_env,
                                          timesteps_per_actorbatch,
                                          stochastic = True,
                                          eval_iters = eval_iters)
        actors.append(newActor)

    seg_gen = traj_segment_generator_eval(pi, base_env, timesteps_per_actorbatch, stochastic=True)
    flatten_weights = pi.get_Flat_variables()()
    opt = cma.CMAOptions()
    opt['tolfun'] = max_fitness
    opt['popsize'] = popsize
    opt['maxiter'] = gensize
    opt['verb_disp'] = 0
    opt['verb_log'] = 0
    opt['seed'] = seed
    opt['AdaptSigma'] = False
    # opt['bounds'] = bounds
    es = cma.CMAEvolutionStrategy(flatten_weights,
                                  sigma, opt)
    costs = None
    best_solution = None

    while True:
        if max_timesteps and timesteps_so_far >= max_timesteps:
            print("Max time steps")
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            print("Max episodes")
            break
        elif max_iters and iters_so_far >= max_iters:
            print("Max iterations")
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            print("Max time")
            break
        elif es.countiter >= opt['maxiter']:
            print("Max generations")
            break
        eval_seg = seg_gen.__next__()
        lrlocal = (eval_seg["ep_lens"], eval_seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        solutions = es.ask()
        if costs is not None:
            solutions[np.argmax(costs)] = np.copy(best_solution)
        ob_segs = None
        segs = []
        costs = []
        lens = []
        for id, solution in enumerate(solutions):
            pi.set_Flat_variables(solution)
            seg = actors[id].__next__()
            costs.append(-np.mean(seg["ep_rets"]))
            lens.append(np.sum(seg["ep_lens"]))
            segs.append(seg)
            if ob_segs is None:
                ob_segs = {'ob': np.copy(seg['ob'])}
            else:
                ob_segs['ob'] = np.append(ob_segs['ob'], seg['ob'], axis=0)
        # Weights decay
        l2_decay = compute_weight_decay(0.999, solutions)
        costs += l2_decay
        costs, real_costs = fitness_normalization(costs)
        es.tell_real_seg(solutions = solutions, function_values = costs, real_f = real_costs, segs = segs)
        best_solution = np.copy(es.result[0])
        best_fitness = -es.result[1]
        print("Generation:", es.countiter)
        print("Best Solution Fitness:", best_fitness)
        pi.set_Flat_variables(best_solution)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob = ob_segs["ob"]
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        seg = seg_gen.__next__()
        iters_so_far += 1
        episodes_so_far += sum(lens)


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis = 1)


def fitness_normalization(x):
    x = np.asarray(x).flatten()
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, x


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
