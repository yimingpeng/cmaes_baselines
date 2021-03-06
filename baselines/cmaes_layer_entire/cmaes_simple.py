import time
from collections import deque

import baselines.common.tf_util as U
import cma
import numpy as np
from baselines import logger
from mpi4py import MPI
import tensorflow as tf
from baselines.common import zipsame


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


def traj_segment_generator(pi, env, horizon, stochastic, eval_iters, seg_gen):
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
        episodes_so_far, tstart,best_fitness
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
          seed = 0
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = base_env.observation_space
    ac_space = base_env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    backup_pi = policy_fn("backup_pi", ob_space,
                          ac_space)  # Construct a network for every individual to adapt during the es evolution


    var_list = pi.get_trainable_variables()
    layer_var_list = []
    for i in range(pi.num_hid_layers):
        layer_var_list.append([v for v in var_list if v.name.split("/")[2].startswith(
                'fc%i' % (i + 1))])
    logstd_var_list = [v for v in var_list if v.name.split("/")[2].startswith(
            "logstd")]
    if len(logstd_var_list) != 0:
        layer_var_list.append([v for v in var_list if v.name.split("/")[2].startswith(
            "final")] + logstd_var_list)

    U.initialize()
    layer_set_operate_list = []
    layer_get_operate_list = []
    for var in layer_var_list:
        layer_set_operate_list.append(U.SetFromFlat(var))
        layer_get_operate_list.append(U.GetFlat(var))

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer, best_fitness
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
    seg_gen = traj_segment_generator_eval(backup_pi, base_env, timesteps_per_actorbatch, stochastic = True)
    actors = []
    for i in range(popsize):
        newActor = traj_segment_generator(pi, base_env,
                                          timesteps_per_actorbatch,
                                          stochastic = True,
                                          eval_iters = eval_iters, seg_gen = seg_gen)
        actors.append(newActor)
    best_fitness = -np.inf
    opt = cma.CMAOptions()
    opt['tolfun'] = max_fitness
    opt['popsize'] = popsize
    opt['maxiter'] = gensize
    opt['verb_disp'] = 0
    opt['verb_log'] = 0
    # opt['seed'] = seed
    opt['AdaptSigma'] = True
    # opt['bounds'] = bounds
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

        # Linearly decay the exploration
        sigma_adapted = max(sigma - float(timesteps_so_far) / max_timesteps, 0)

        logger.log("********** Iteration %i ************" % iters_so_far)
        eval_seg = seg_gen.__next__()
        rewbuffer.extend(eval_seg["ep_rets"])
        lenbuffer.extend(eval_seg["ep_lens"])
        if iters_so_far == 0:
            result_record()

        for i in range(len(layer_var_list)):
            assign_backup_eq_new()  # backup current policy
            logger.log("Current Layer:"+ str(layer_var_list[i]))
            flatten_weights = layer_get_operate_list[i]()
            es = cma.CMAEvolutionStrategy(flatten_weights,
                                          sigma, opt)
            costs = None
            best_solution = None

            die_out_count = 0
            while True:
                if es.countiter >= gensize:
                    logger.log("Max generations for current layer")
                    break
                solutions = es.ask()
                ob_segs = None
                segs = []
                costs = []
                lens = []
                for id, solution in enumerate(solutions):
                    layer_set_operate_list[i](solution)
                    seg = actors[id].__next__()
                    costs.append(-np.mean(seg["ep_rets"]))
                    lens.append(np.sum(seg["ep_lens"]))
                    segs.append(seg)
                    if ob_segs is None:
                        ob_segs = {'ob': np.copy(seg['ob'])}
                    else:
                        ob_segs['ob'] = np.append(ob_segs['ob'], seg['ob'], axis = 0)
                    assign_new_eq_backup()
                # Weights decay
                l2_decay = compute_weight_decay(0.01, solutions)
                costs += l2_decay
                costs, real_costs = fitness_normalization(costs)
                es.tell_real_seg(solutions = solutions, function_values = costs, real_f = real_costs, segs = segs)
                best_solution = np.copy(es.result[0])
                best_fitness = -es.result[1]
                rewbuffer.extend(es.result[3]["ep_rets"])
                lenbuffer.extend(es.result[3]["ep_lens"])
                layer_set_operate_list[i](best_solution)
                logger.log("Update the layer")
                logger.log("Generation:", es.countiter)
                logger.log("Best Solution Fitness:", best_fitness)

                ob = ob_segs["ob"]
                if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for observation normalization
                episodes_so_far += sum(lens)
            es = None
            import gc
            gc.collect()
        iters_so_far += 1


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
