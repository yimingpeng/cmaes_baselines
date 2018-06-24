import time
from collections import deque

import numpy as np
import tensorflow as tf
from mpi4py import MPI

import baselines.common.tf_util as U
import cma
from baselines import logger
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines.common.mpi_adam import MpiAdam
from baselines.ppo_cmaes.cnn_policy import CnnPolicy

test_rewbuffer = deque(maxlen = 100)  # test buffer for episode rewards
KL_Condition = False
mean_action_Condition = True


def traj_segment_generator_eval(pi, env, horizon, stochastic):
    t = 0
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    while True:
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        ob, rew, new, _ = env.step(ac)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def traj_segment_generator(pi, env, horizon, stochastic, eval_gen):
    # Trajectories generators
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
    next_obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    traj_index = []
    index_count = 0

    while True:
        if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
            result_record(eval_gen)
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "next_ob": next_obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "traj_index": traj_index}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            index_count = 0
            traj_index = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        next_obs[i] = ob

        cur_ep_ret += rew
        cur_ep_len += 1
        timesteps_so_far += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            traj_index.append(index_count)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1
        index_count += 1


def result_record(seg_gen):
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
    eval_seg = seg_gen.__next__()
    lrlocal = (eval_seg["ep_lens"], eval_seg["ep_rets"])  # local values
    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
    lens, rews = map(flatten_lists, zip(*listoflrpairs))
    lenbuffer.extend(lens)
    rewbuffer.extend(rews)
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


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return weight_decay * np.mean(model_param_grid * model_param_grid, axis = 1)


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"],
                    0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, test_env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          # CMAES
          max_fitness,  # has to be negative, as cmaes consider minization
          popsize,
          gensize,
          bounds,
          sigma,
          eval_iters,
          max_v_train_iter,
          max_timesteps = 0, max_episodes = 0, max_iters = 0, max_seconds = 0,
          # time constraint
          callback = None,
          # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon = 1e-5,
          schedule = 'constant',
          # annealing for stepsize parameters (epsilon and adam)
          seed,
          env_id
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy
    backup_pi = policy_fn("backup_pi", ob_space,
                          ac_space)  # Construct a network for every individual to adapt during the es evolution
    pi_zero = policy_fn("zero_pi", ob_space, ac_space)  # pi_0 will only be updated along with iterations

    reward = tf.placeholder(dtype = tf.float32, shape = [None])  # step rewards
    atarg = tf.placeholder(dtype = tf.float32, shape = [None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype = tf.float32, shape = [None])  # Empirical return

    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name = "ob")
    next_ob = U.get_placeholder_cached(name = "next_ob")  # next step observation for updating q function
    ac = U.get_placeholder_cached(name = "act")  # action placeholder for computing q function

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent


    pi_adv = pi.qpred - pi.vpred
    adv_mean, adv_var = tf.nn.moments(pi_adv, axes = [0])
    normalized_pi_adv = (pi_adv - adv_mean) / tf.sqrt(adv_var)

    qf_loss = tf.reduce_mean(tf.square(reward + gamma * pi.vpred - pi.qpred))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    qf_losses = [qf_loss]
    vf_losses = [vf_loss]
    pol_loss = -tf.reduce_mean(normalized_pi_adv)

    # Advantage function should be improved
    losses = [pol_loss, pol_entpen, meankl, meanent]
    loss_names = ["pol_surr_2", "pol_entpen", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    qf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "qf")]
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol")]

    vf_lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])

    qf_lossandgrad = U.function([ob, ac, next_ob, lrmult, reward],
                                qf_losses + [U.flatgrad(qf_loss, qf_var_list)])

    qf_adam = MpiAdam(qf_var_list, epsilon = adam_epsilon)

    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)

    assign_old_eq_new = U.function([], [], updates = [tf.assign(oldv, newv)
                                                      for (oldv, newv) in zipsame(
            oldpi.get_variables(), pi.get_variables())])

    assign_backup_eq_new = U.function([], [], updates = [tf.assign(backup_v, newv)
                                                         for (backup_v, newv) in zipsame(
            backup_pi.get_variables(), pi.get_variables())])
    assign_new_eq_backup = U.function([], [], updates = [tf.assign(newv, backup_v)
                                                         for (newv, backup_v) in zipsame(
            pi.get_variables(), backup_pi.get_variables())])
    # Compute all losses

    mean_pi_actions = U.function([ob], [pi.pd.mode()]) # later for computing pol_loss
    compute_pol_losses = U.function([ob, next_ob, ac],[pol_loss])

    U.initialize()

    get_pi_flat_params = U.GetFlat(pol_var_list)
    set_pi_flat_params = U.SetFromFlat(pol_var_list)

    vf_adam.sync()
    qf_adam.sync()

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer, tstart, ppo_timesteps_so_far, best_fitness

    episodes_so_far = 0
    timesteps_so_far = 0
    ppo_timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards

    best_fitness = np.inf

    eval_gen = traj_segment_generator_eval(pi, test_env, timesteps_per_actorbatch, stochastic = True)  # For evaluation
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic = True,
                                     eval_gen = eval_gen)  # For train V Func

    # Build generator for all solutions
    actors = []
    best_fitness = 0
    for i in range(popsize):
        newActor = traj_segment_generator(pi, env,
                                          timesteps_per_actorbatch,
                                          stochastic = True, eval_gen = eval_gen)
        actors.append(newActor)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

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

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)

        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        # Generate new samples
        # Train V func
        for i in range(max_v_train_iter):
            logger.log("Iteration:" + str(iters_so_far) + " - sub-train iter for V func:" + str(i))
            logger.log("Generate New Samples")
            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            ob, ac, next_ob, atarg, reward, tdlamret, traj_idx = seg["ob"], seg["ac"], seg["next_ob"], seg["adv"], seg["rew"], seg["tdlamret"], \
                                                        seg["traj_index"]
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            d = Dataset(dict(ob = ob, ac = ac, atarg = atarg, vtarg = tdlamret), shuffle = not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for normalization

            assign_old_eq_new()  # set old parameter values to new parameter values
            # Train V function
            logger.log("Training V Func and Evaluating V Func Losses")
            for _ in range(optim_epochs):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *vf_losses, g = vf_lossandgrad( batch["ob"],batch["ac"], batch["atarg"], batch["vtarg"],
                                                   cur_lrmult)
                    vf_adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(vf_losses)
                logger.log(fmt_row(13, np.mean(losses, axis = 0)))

            d_q = Dataset(dict(ob = ob, ac = ac, next_ob = next_ob, reward = reward,
                               atarg = atarg, vtarg = tdlamret), shuffle = not pi.recurrent)

            # Re-train q function
            logger.log("Training Q Func Evaluating Q Func Losses")
            for _ in range(optim_epochs):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d_q.iterate_once(optim_batchsize):
                    *qf_losses, g = qf_lossandgrad(batch["next_ob"],  batch["ac"], batch["ob"],
                                                   cur_lrmult, batch["reward"])
                    qf_adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(qf_losses)
                logger.log(fmt_row(13, np.mean(losses, axis = 0)))


        # CMAES Train Policy
        assign_old_eq_new()  # set old parameter values to new parameter values
        assign_backup_eq_new()  # backup current policy
        flatten_weights = get_pi_flat_params()
        opt = cma.CMAOptions()
        opt['tolfun'] = max_fitness
        opt['popsize'] = popsize
        opt['maxiter'] = gensize
        opt['verb_disp'] = 0
        opt['verb_log'] = 0
        opt['seed'] = seed
        opt['AdaptSigma'] = True
        es = cma.CMAEvolutionStrategy(flatten_weights,
                                      sigma, opt)
        while True:
            if es.countiter >= gensize:
                logger.log("Max generations for current layer")
                break
            logger.log("Iteration:" + str(iters_so_far) + " - sub-train Generation for Policy:" + str(es.countiter))
            logger.log("Sigma=" + str(es.sigma))
            solutions = es.ask()
            costs = []
            lens = []

            assign_backup_eq_new()  # backup current policy

            for id, solution in enumerate(solutions):
                set_pi_flat_params(solution)
                losses = []
                cost = compute_pol_losses(ob, ob, mean_pi_actions(ob)[0])
                costs.append(cost[0])
                assign_new_eq_backup()
            # Weights decay
            l2_decay = compute_weight_decay(0.99, solutions)
            costs += l2_decay
            # costs, real_costs = fitness_normalization(costs)
            costs, real_costs = fitness_rank(costs)
            es.tell_real_seg(solutions = solutions, function_values = costs, real_f = real_costs, segs = None)
            best_solution = es.result[0]
            best_fitness = es.result[1]
            logger.log("Best Solution Fitness:" + str(best_fitness))
            set_pi_flat_params(best_solution)

        iters_so_far += 1
        episodes_so_far += sum(lens)


def fitness_rank(x):
    x = np.asarray(x).flatten()
    ranks = np.empty(len(x))
    ranks[x.argsort()] = np.arange(len(x))
    ranks /= (len(x) - 1)
    ranks -= .5
    return ranks, x



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
