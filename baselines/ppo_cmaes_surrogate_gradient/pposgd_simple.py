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


def traj_segment_generator(pi, env, horizon, stochastic):
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
            result_record()
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
        index_count  += 1


def result_record():
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
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


def learn(env, policy_fn, *,
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
    backup_pi = policy_fn("backup_pi", ob_space, ac_space)  # Construct a network for every individual to adapt during the es evolution

    pi_params = tf.placeholder(dtype = tf.float32, shape = [None])
    old_pi_params = tf.placeholder(dtype = tf.float32, shape = [None])
    atarg = tf.placeholder(dtype = tf.float32, shape = [None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype = tf.float32, shape = [None])  # Empirical return

    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    vf_losses = [vf_loss]

    # Absolute mean action error
    param_dist = tf.reduce_sum(tf.square(pi_params - old_pi_params))
    mean_action_loss = tf.cast(tf.reduce_mean(tf.abs(pi.pd.mode() - oldpi.pd.mode())), tf.float32)

    pol_losses = [pol_surr + pol_entpen + param_dist + mean_action_loss]
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    # print(var_list)
    if isinstance(pi, CnnPolicy):
        lin_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
            "lin")]
        vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
            "logits")]
        pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
            "value")]
        # Policy + Value function, the final layer, all trainable variables
        # Remove vf variables
        var_list = lin_var_list + pol_var_list
    else:
        fc2_var_list = [v for v in var_list if v.name.split("/")[2].startswith(
            "fc2")]
        final_var_list = [v for v in var_list if v.name.split("/")[
            2].startswith(
            "final")]
        # var_list = vf_var_list + pol_var_list
        var_list = fc2_var_list + final_var_list
    # print(var_list)
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol")]

    old_pi_var_list = oldpi.get_trainable_variables()
    if isinstance(oldpi, CnnPolicy):
        lin_var_list = [v for v in old_pi_var_list if v.name.split("/")[1].startswith(
            "lin")]
        vf_var_list = [v for v in old_pi_var_list if v.name.split("/")[1].startswith(
            "logits")]
        pol_var_list = [v for v in old_pi_var_list if v.name.split("/")[1].startswith(
            "value")]
        # Policy + Value function, the final layer, all trainable variables
        # Remove vf variables
        old_pi_var_list = lin_var_list + pol_var_list
    else:
        fc2_var_list = [v for v in old_pi_var_list if v.name.split("/")[2].startswith(
            "fc2")]
        final_var_list = [v for v in old_pi_var_list if v.name.split("/")[
            2].startswith(
            "final")]
        # var_list = vf_var_list + pol_var_list
        old_pi_var_list = fc2_var_list + final_var_list
    old_pi_pol_var_list = [v for v in old_pi_var_list if v.name.split("/")[1].startswith(
        "pol")]
    # compute the Advantage estimations: A = Q - V for pi
    # get_A_estimation = U.function([ob, ob, ac], [pi.qpred - pi.vpred])
    # get_A_pi_zero_estimation = U.function([ob, ob, ac], [pi_zero.qpred - pi_zero.vpred])
    # compute the Advantage estimations: A = Q - V for evalpi

    # compute the mean action for given states under pi
    # mean_pi_actions = U.function([ob], [pi.pd.mode()])
    # mean_pi_zero_actions = U.function([ob], [pi_zero.pd.mode()])

    vf_lossandgrad = U.function([ob, ret, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])
    #
    # qf_lossandgrad = U.function([ob, ac, next_ob, mean_ac, lrmult, reward],
    #                             qf_losses + [U.flatgrad(qf_loss, qf_var_list)])
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                             losses + [U.flatgrad(total_loss, var_list)])

    # qf_adam = MpiAdam(qf_var_list, epsilon = adam_epsilon)

    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)
    adam = MpiAdam(var_list, epsilon = adam_epsilon)

    # assign_target_q_eq_eval_q = U.function([], [], updates = [tf.assign(target_q, eval_q)
    #                                                           for (target_q, eval_q) in zipsame(
    #         mean_qf_var_list, qf_var_list)])
    assign_old_eq_new = U.function([], [], updates = [tf.assign(oldv, newv)
                                                      for (oldv, newv) in zipsame(
            oldpi.get_variables(), pi.get_variables())])

    assign_backup_eq_new = U.function([], [], updates = [tf.assign(backup_v, newv)
                                                      for (backup_v, newv) in zipsame(
            backup_pi.get_trainable_variables(), pi.get_trainable_variables())])
    assign_new_eq_backup = U.function([], [], updates = [tf.assign(newv, backup_v)
                                                      for (newv, backup_v) in zipsame(
            pi.get_trainable_variables(), backup_pi.get_trainable_variables())])
    # Compute all losses
    #
    compute_v_losses = U.function([ob, ac, atarg, ret, lrmult], vf_losses)
    # compute_q_losses = U.function([ob, ac, next_ob, lrmult, reward], qf_losses)
    compute_pol_losses = U.function([ob, ac, atarg, ret, lrmult, pi_params, old_pi_params], pol_losses)
    # compute_losses = U.function([ob, ac, atarg, ret, lrmult], [total_loss])

    get_pi_traj_prob = U.function([ob, ac], [tf.exp(pi.pd.logp(ac))])
    get_old_pi_traj_prob = U.function([ob, ac], [tf.exp(oldpi.pd.logp(ac))])

    U.initialize()

    get_pi_flat_params = U.GetFlat(pol_var_list)
    set_pi_flat_params = U.SetFromFlat(pol_var_list)
    get_old_pi_flat_params = U.GetFlat(old_pi_pol_var_list)
    # set_pi_pol_flat_params = U.SetFromFlat(pol_var_list)

    # vf_adam.sync()
    # qf_adam.sync()
    adam.sync()

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer, tstart, ppo_timesteps_so_far, best_fitness

    episodes_so_far = 0
    timesteps_so_far = 0
    ppo_timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards

    test_rew_buffer = []
    best_fitness = np.inf

    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic = True)  # For train
    eval_gen = traj_segment_generator_eval(pi, env, timesteps_per_actorbatch, stochastic = True)  # For train

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    assign_backup_eq_new() # backup current policy
    flatten_weights = get_pi_flat_params()
    opt = cma.CMAOptions()
    opt['tolfun'] = max_fitness
    opt['popsize'] = popsize
    opt['maxiter'] = gensize
    opt['verb_disp'] = 0
    opt['verb_log'] = 0
    opt['seed'] = seed
    opt['AdaptSigma'] = True
    opt['bounds'] = bounds
    # sigma1 = sigma - 0.01 * iters_so_far
    # if sigma1 < 0.0001:
    #     sigma1 = 0.0001
    # print("Sigma=", sigma1)
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

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Generation %i ************" % iters_so_far)
        logger.log("simga:" + str(es.sigma))

        #Generate new samples
        logger.log("Generate New Samples")
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, next_ob, ac, reward, tdlamret = seg["ob"], seg["next_ob"], seg["ac"], seg["rew"], seg["tdlamret"]
        # d_v = Dataset(dict(ob = ob, vtarg = tdlamret), shuffle = not pi.recurrent)
        # optim_batchsize = optim_batchsize or ob.shape[0]
        ob, ac, atarg, reward, tdlamret, traj_idx = seg["ob"], seg["ac"], seg["adv"],  seg["rew"],seg["tdlamret"], seg["traj_index"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)  # Record the train results
        rewbuffer.extend(rews)

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for normalization

        # Train V function
        logger.log("Training V Func and Evaluating V Func Losses")
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *vf_losses, g = vf_lossandgrad(batch["ob"], batch["vtarg"],
                                               cur_lrmult)
                vf_adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(vf_losses)
            logger.log(fmt_row(13, np.mean(losses, axis = 0)))
        # sigma1 = sigma - 0.001 * iters_so_far
        # if sigma1 < 0.00001:
        #     sigma1 = 0.00001
        print("cur_lrmult=", cur_lrmult)
        # solutions = es.ask(sigma_fac = cur_lrmult)
        solutions = es.ask()
        costs = []
        lens = []

        assign_old_eq_new() # set old parameter values to new parameter values
        assign_backup_eq_new() # backup current policy

        # for id, solution in enumerate(solutions):
        #     set_pi_flat_params(solution)
        #     i = 0
        #     abs_act_dist = np.mean(np.abs(mean_pi_actions(ob)[0] - mean_old_pi_actions(ob)[0]))
        #     while abs_act_dist > 0.05:
        #         i += 1
        #         solutions[id] = es.ask(number = 1, sigma_fac = 0.999 ** i)[0]
        #         set_pi_flat_params(solutions[id])
        #         abs_act_dist = np.mean(np.abs(mean_pi_actions(ob)[0] - mean_old_pi_actions(ob)[0]))
        #         logger.log("Regenerate Solution for " + str(i) + " times for ID:" + str(
        #             id) + " mean_action_dist:" + str(abs_act_dist))
        assign_new_eq_backup()  # Restore the backup after all feasible solutions are generated

        for id, solution in enumerate(solutions):
            # pi.set_Flat_variables(solution)
            set_pi_flat_params(solution)
            cost = compute_pol_losses(ob, ac, atarg, tdlamret, cur_lrmult, get_pi_flat_params(), get_old_pi_flat_params())
            # idx0 = 0
            # i = 0
            # performances = []
            # for idx in traj_idx:
            #     pi_prob_traj = np.prod(get_pi_traj_prob(ob[range(idx0, idx)], ac[range(idx0, idx)])[0])
            #     old_pi_prob_traj = np.prod(get_old_pi_traj_prob(ob[range(idx0, idx)], ac[range(idx0, idx)])[0])
            #     if old_pi_prob_traj == 0:
            #         old_pi_prob_traj += 1e-08
            #     performances.append((pi_prob_traj/old_pi_prob_traj) * tdlamret[i])
            #     idx0 = idx+1
            #     i += 1
            # costs.append(cost[0] + np.mean(np.abs(mean_pi_actions(ob)[0] - mean_old_pi_actions(ob)[0])))
            costs.append(cost[0])
            assign_new_eq_backup()
        # Weights decay
        l2_decay = compute_weight_decay(0.99, solutions)
        costs += l2_decay
        # costs, real_costs = fitness_normalization(costs)
        costs, real_costs = fitness_rank(costs)
        es.tell_real_seg(solutions = solutions, function_values = costs, real_f = real_costs, segs = None)
        best_solution =es.result[0]
        best_fitness = es.result[1]
        # print("Generation:", es.countiter)
        logger.log("Best Solution Fitness:", best_fitness)
        set_pi_flat_params(best_solution)

        eval_pi_seg = eval_gen.__next__()
        logger.log("Current Test Performance:" + str(np.mean(eval_pi_seg["ep_rets"])))

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        iters_so_far += 1
        episodes_so_far += sum(lens)


# def fitness_normalization(x):
#     x = np.asarray(x).flatten()
#     mean = np.mean(x)
#     std = np.std(x)
#     return (x - mean) / std, x
#
#
def fitness_rank(x):
    x = np.asarray(x).flatten()
    ranks = np.empty(len(x))
    ranks[x.argsort()] = np.arange(len(x))
    ranks /= (len(x) - 1)
    ranks -= .5
    return ranks, x


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def fitness_normalization(x):
    x = np.asarray(x).flatten()
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, x
