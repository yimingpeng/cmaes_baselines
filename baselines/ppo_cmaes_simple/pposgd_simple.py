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
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
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
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


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
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis = 1)


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
    backup_pi = policy_fn("backup_pi", ob_space, ac_space)  # Network for cmaes individual to train
    pi_zero = policy_fn("zero_pi", ob_space, ac_space)  # pi_0 will only be updated along with iterations

    atarg = tf.placeholder(dtype = tf.float32, shape = [None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype = tf.float32, shape = [None])  # Empirical return

    reward = tf.placeholder(dtype = tf.float32, shape = [None])  # step rewards
    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name = "ob")
    next_ob = U.get_placeholder_cached(name = "next_ob")  # next step observation for updating q function
    ac = U.get_placeholder_cached(name = "act")  # action placeholder for computing q function
    mean_ac = U.get_placeholder_cached(name = "mean_act")  # action placeholder for computing q function

    # qf_loss = tf.reduce_mean(tf.square( - pi.qpred))
    # td_error = pi.qpred - reward + gamma * pi.mean_qpred
    # errors = U.huber_loss(td_error)
    qf_loss = tf.reduce_mean(tf.square(reward + gamma * pi.mean_qpred - pi.qpred))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    qf_losses = [qf_loss]
    vf_losses = [vf_loss]

    var_list = pi.get_trainable_variables()
    qf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "qf")]
    mean_qf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "meanqf")]
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol")]

    # compute the Advantage estimations: A = Q - V for pi
    get_A_estimation = U.function([ob, ob, ac], [pi.qpred - pi.vpred])
    get_A_pi_zero_estimation = U.function([ob, ob, ac], [pi_zero.qpred - pi_zero.vpred])
    # compute the Advantage estimations: A = Q - V for evalpi

    # compute the mean action for given states under pi
    mean_pi_actions = U.function([ob], [pi.pd.mode()])
    mean_pi_zero_actions = U.function([ob], [pi_zero.pd.mode()])

    vf_lossandgrad = U.function([ob, ret, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])

    qf_lossandgrad = U.function([ob, ac, next_ob, mean_ac, lrmult, reward],
                                qf_losses + [U.flatgrad(qf_loss, qf_var_list)])

    qf_adam = MpiAdam(qf_var_list, epsilon = adam_epsilon)

    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)

    assign_target_q_eq_eval_q = U.function([], [], updates = [tf.assign(target_q, eval_q)
                                                              for (target_q, eval_q) in zipsame(
            mean_qf_var_list, qf_var_list)])
    assign_old_eq_new = U.function([], [], updates = [tf.assign(oldv, newv)
                                                      for (oldv, newv) in zipsame(
            oldpi.get_variables(), pi.get_variables())])

    # Assign pi to backup (only backup trainable variables)
    assign_backup_eq_new = U.function([], [], updates = [tf.assign(backup_v, newv)
                                                         for (backup_v, newv) in zipsame(
            backup_pi.get_variables(), pi.get_variables())])

    # Assign backup back to pi
    assign_new_eq_backup = U.function([], [], updates = [tf.assign(newv, backup_v)
                                                         for (newv, backup_v) in zipsame(
            pi.get_variables(), backup_pi.get_variables())])

    # Assign pi to pi0 (for parameter updating constraints)
    assign_pi_zero_eq_new = U.function([], [], updates = [tf.assign(pi_zero_v, newv)
                                                          for (pi_zero_v, newv) in zipsame(
            pi_zero.get_variables(), pi.get_variables())])

    # Compute all losses
    #
    # compute_v_losses = U.function([ob, ac, atarg, ret, lrmult], vf_losses)
    # compute_q_losses = U.function([ob, ac, next_ob, lrmult, reward], qf_losses)

    U.initialize()

    get_pi_pol_flat_params = U.GetFlat(pol_var_list)
    set_pi_pol_flat_params = U.SetFromFlat(pol_var_list)

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

    test_rew_buffer = []

    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic = True)  # For train
    eval_pi_zero_gen = traj_segment_generator_eval(pi_zero, env, 4096, stochastic = True)  # For test
    eval_pi_gen = traj_segment_generator_eval(pi, env, 4096, stochastic = True)  # For Test

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    i = 0
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        # PPO Train V and Q
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        ob, next_ob, ac, reward, tdlamret = seg["ob"], seg["next_ob"], seg["ac"], seg["rew"], seg["tdlamret"]
        d_v = Dataset(dict(ob = ob, vtarg = tdlamret), shuffle = not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)  # Record the train results
        rewbuffer.extend(rews)

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for normalization

        # Re-train V function
        logger.log("Training V Func and Evaluating V Func Losses")
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d_v.iterate_once(optim_batchsize):
                *vf_losses, g = vf_lossandgrad(batch["ob"], batch["vtarg"],
                                               cur_lrmult)
                vf_adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(vf_losses)
            logger.log(fmt_row(13, np.mean(losses, axis = 0)))

        # logger.log("Training Q Func Evaluating Q Func Losses")
        # d_q = Dataset(dict(ob = ob,
        #                    next_ob = next_ob,
        #                    ac = ac,
        #                    vtarg = tdlamret,
        #                    reward=reward,
        #                    mean_actions=mean_pi_actions(ob)[0]),
        #                    shuffle = not pi.recurrent)

        # Random select transitions to train Q
        # for _ in range(optim_epochs):
        #     losses = []  # list of tuples, each of which gives the loss for a minibatch
        #     for batch in d_q.iterate_once(optim_batchsize):
        #         *qf_losses, g = qf_lossandgrad(batch["ob"], batch["ac"], batch["next_ob"],
        #                                        batch["mean_actions"],
        #                                        cur_lrmult, batch["reward"])
        #         qf_adam.update(g, optim_stepsize * cur_lrmult)
        #         losses.append(qf_losses)
        #     logger.log(fmt_row(13, np.mean(losses, axis = 0)))

        random_idx = []
        len_repo = len(ob)
        optim_epochs_q = int(len_repo / optim_batchsize) if int(
            len_repo / optim_batchsize) > optim_epochs else optim_epochs
        for _ in range(optim_epochs_q):
            random_idx.append(np.random.choice(range(len_repo), optim_batchsize))

        # Re-train q function
        logger.log("Training Q Func Evaluating Q Func Losses")
        for _ in range(optim_epochs_q):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for idx in random_idx:
                *qf_losses, g = qf_lossandgrad(ob[idx], ac[idx], next_ob[idx],
                                               mean_pi_actions(ob)[0][idx],
                                               cur_lrmult, reward[idx])
                qf_adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(qf_losses)
            logger.log(fmt_row(13, np.mean(losses, axis = 0)))

        assign_target_q_eq_eval_q()

        # if iters_so_far != 0 and iters_so_far % 5 == 0:
        if iters_so_far % 1 == 0:
            logger.log("CMAES Policy Optimization")
            # Make two q network equal

            # Record Pi0 behavior 25 times
            assign_pi_zero_eq_new()  # Record the pi_0, initially it equals to pi
            eval_pi_zero_seg = eval_pi_zero_gen.__next__()
            test_rew_buffer.append(eval_pi_zero_seg["ep_rets"])  # Record the pi zero results

            # CMAES
            policy_weights = get_pi_pol_flat_params()

            opt = cma.CMAOptions()
            opt['tolfun'] = max_fitness
            opt['popsize'] = popsize
            opt['maxiter'] = gensize
            opt['verb_disp'] = 0
            opt['verb_log'] = 0
            opt['seed'] = seed
            opt['AdaptSigma'] = True
            # opt['bounds'] = bounds
            # sigma1 = sigma - 0.001 * iters_so_far
            # if sigma1 < 0.0001:
            #     sigma1 = 0.0001
            # # print("Sigma=", sigma1)
            # es = cma.CMAEvolutionStrategy(policy_weights,
            #                               sigma1, opt)
            es = cma.CMAEvolutionStrategy(policy_weights,
                                          sigma, opt)
            best_fitness = -np.inf
            costs = None
            while True:
                if es.countiter >= opt['maxiter']:
                    break
                solutions = es.ask()
                assign_backup_eq_new()  # backup current policy, after Q and V have been trained
                if mean_action_Condition:
                    for id, solution in enumerate(solutions):
                        set_pi_pol_flat_params(solution)
                        i = 0
                        abs_act_dist = np.mean(np.abs(mean_pi_actions(ob)[0] - mean_pi_zero_actions(ob)[0]))
                        while abs_act_dist > 0.05:
                            i += 1
                            solutions[id] = es.ask(number = 1, sigma_fac = 0.999 ** i)[0]
                            set_pi_pol_flat_params(solutions[id])
                            abs_act_dist = np.mean(np.abs(mean_pi_actions(ob)[0] - mean_pi_zero_actions(ob)[0]))
                            logger.log("Regenerate Solution for " + str(i) + " times for ID:" + str(
                                id) + " mean_action_dist:" + str(abs_act_dist))
                    assign_new_eq_backup()  # Restore the backup after all feasible solutions are generated

                segs = []
                ob_segs = None
                costs = []
                lens = []
                # Evaluation
                a_func = get_A_estimation(ob, ob, mean_pi_actions(ob)[0])
                a_func_pi_zero = get_A_pi_zero_estimation(ob, ob, mean_pi_zero_actions(ob)[0])
                logger.log("A-pi-zero:" + str(np.mean(a_func_pi_zero)))
                logger.log("A-pi-best:" + str(np.mean(a_func)))
                # Evaluation
                for id, solution in enumerate(solutions):
                    set_pi_pol_flat_params(solution)
                    new_a_func = get_A_estimation(ob, ob, mean_pi_actions(ob)[0])
                    logger.log("A-pi" + str(id + 1) + ":" + str(np.mean(new_a_func)))
                    coeff1 = 0.9
                    coeff2 = 0.9
                    cost = - (np.mean(new_a_func))
                    costs.append(cost)
                    assign_new_eq_backup()  # Restore the backup
                # costs = fitness_rank(fitness_rank)
                es.tell_real_seg(solutions = solutions, function_values = costs, real_f = costs, segs = None)
                # if -min(costs) >= np.mean(a_func):
                if es.result[1] > best_fitness:
                    logger.log("Update Policy by CMAES due to current best["
                          + str(es.result[1]) + "] >= global best[" + str(best_fitness) + "]")
                    best_solution = es.result[0]
                    best_fitness = es.result[1]
                    set_pi_pol_flat_params(best_solution)
                logger.log("Generation:", es.countiter)
                # eval_pi_seg = eval_pi_gen.__next__()
                # test_rew_buffer.append(eval_pi_seg["ep_rets"])
                # logger.log("Current Evaluation - Mean {0} Std {1}".format(np.mean(eval_pi_seg["ep_rets"]),
                #                                                           np.std(eval_pi_seg["ep_rets"])))
                # set old parameter values to new parameter values
                # break
            # Record CMAES-Updated Pi behaviors 25 times
            eval_pi_seg = eval_pi_gen.__next__()
            test_rew_buffer.append(eval_pi_seg["ep_rets"])
            assign_pi_zero_eq_new()  # Update the p0

            logger.log("Pi 0 Mean {0} Std {1}".format(np.mean(test_rew_buffer[0]),
                                                              np.std(test_rew_buffer[0])))
            logger.log("Pi 1 Mean {0} Std {1}".format(np.mean(test_rew_buffer[1]),
                                                               np.std(test_rew_buffer[1])))
            test_rew_buffer.clear()
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
