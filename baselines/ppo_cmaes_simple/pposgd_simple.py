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
    logger.log("********** Iteration %i ************" % iters_so_far)
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


#TODO: Modify
def uniform_select(weight, num_of_weights):
    # Temporarily disabled uniform selection, essentially this is just pure index range from 0 to the maximimum
    length = len(weight) if len(weight) < num_of_weights else \
        num_of_weights
    # index = np.random.choice(range(len(
    #     weight)), length, replace = False)
    index = range(length)
    return index, np.take(weight, index)


def set_uniform_weights(original_weight, new_weight, index):
    result_weight = original_weight
    np.put(result_weight, index, new_weight)
    return result_weight


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
    pi_zero = policy_fn("zero_pi", ob_space, ac_space)  # Network for cmaes individual to train

    atarg = tf.placeholder(dtype = tf.float32, shape = [
        None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype = tf.float32, shape = [None])  # Empirical return

    reward = tf.placeholder(dtype = tf.float32, shape = [None])  # step rewards
    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name = "ob")
    next_ob = U.get_placeholder_cached(name = "next_ob")  # next step observation for updating q function
    ac = U.get_placeholder_cached(name = "act")  # action placeholder for computing q function
    mean_ac = U.get_placeholder_cached(name = "mean_act")  # action placeholder for computing q function

    kloldnew = oldpi.pd.kl(pi.pd)
    klpi_pi_zero = pi.pd.kl(pi_zero.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param,
                             1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(
        tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    y = reward + gamma * pi.mean_qpred
    qf_loss = tf.reduce_mean(tf.square(y - pi.qpred))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen  # v function is independently trained
    qf_losses = [qf_loss]
    vf_losses = [vf_loss]
    qv_losses = [qf_loss, vf_loss]
    losses = [pol_surr, pol_entpen, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "kl", "ent"]

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
        print(var_list)
    # print(var_list)
    qf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "qf")]
    mean_qf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "meanqf")]
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]

    # compute the Advantage estimations: A = Q - V for pi
    get_A_estimation = U.function([ob, next_ob, ac], [pi.qpred - pi.vpred])
    get_A_pi_zero_estimation = U.function([ob, next_ob, ac], [pi_zero.qpred - pi_zero.vpred])
    # compute the Advantage estimations: A = Q - V for evalpi

    # compute the mean action for given states under pi
    mean_pi_actions = U.function([ob], [pi.pd.mode()])
    mean_pi_zero_actions = U.function([ob], [pi_zero.pd.mode()])
    # compute the mean kl
    mean_Kl = U.function([ob], [tf.reduce_mean(klpi_pi_zero)])

    qf_lossandgrad = U.function([ob, ac, next_ob, mean_ac, lrmult, reward],
                                qf_losses + [U.flatgrad(qf_loss, qf_var_list)])
    vf_lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                             losses + [U.flatgrad(total_loss, var_list)])

    qf_adam = MpiAdam(qf_var_list, epsilon = adam_epsilon)

    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)

    adam = MpiAdam(var_list, epsilon = adam_epsilon)

    assign_target_q_eq_eval_q = U.function([], [], updates = [tf.assign(oldqv, newqv)
                                                      for (oldqv, newqv) in zipsame(
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

    compute_v_losses = U.function([ob, ac, atarg, ret, lrmult], vf_losses)
    compute_v_losses = U.function([ob, ac, next_ob, lrmult, reward], qf_losses)
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    pi_set_flat = U.SetFromFlat(pi.get_trainable_variables())
    pi_get_flat = U.GetFlat(pi.get_trainable_variables())
    backup_pi_get_flat = U.GetFlat(backup_pi.get_trainable_variables())
    pi_zero_get_flat = U.GetFlat(pi_zero.get_trainable_variables())

    adam.sync()

    global timesteps_so_far, episodes_so_far, iters_so_far, \
        tstart, lenbuffer, rewbuffer, tstart, ppo_timesteps_so_far, best_fitness
    episodes_so_far = 0
    timesteps_so_far = 0
    ppo_timesteps_so_far = 0
    # cmaes_timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards

    test_rew_buffer = []
    # Prepare for rollouts
    # ----------------------------------------
    # assign pi to eval_pi
    actors = []
    best_fitness = 0

    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic = True)
    eval_gen = traj_segment_generator_eval(pi, env, 4096, stochastic = True)

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

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, next_ob, ac, atarg, tdlamret, reward = seg["ob"], seg["next_ob"], seg["ac"], seg["adv"], seg[
            "tdlamret"], seg["rew"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob = ob, ac = ac, atarg = atarg, vtarg = tdlamret),
                    shuffle = not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        # Record Pi0 behavior 25 times
        eval_seg = eval_gen.__next__()
        test_rew_buffer.append(eval_seg["ep_rets"])

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        # Re-train V function
        logger.log("Evaluating V Func Losses")
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *vf_losses, g = vf_lossandgrad(batch["ob"], batch["ac"],
                                               batch["atarg"], batch["vtarg"],
                                               cur_lrmult)
                vf_adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(vf_losses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        # Random select tansitions to train Q
        random_idx = []
        len_repo = len(seg["ob"])
        optim_epochs_q = int(len_repo / optim_batchsize) if int(len_repo / optim_batchsize) > optim_epochs else optim_epochs
        for _ in range(optim_epochs_q):
            random_idx.append(np.random.choice(range(len_repo), optim_batchsize))

        # Re-train q function
        logger.log("Evaluating Q Func Losses")
        for _ in range(optim_epochs_q):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for idx in random_idx:
                *qf_losses, g = qf_lossandgrad(seg["ob"][idx], seg["ac"][idx], seg["next_ob"][idx], mean_pi_actions(ob)[0][idx],
                                               cur_lrmult, seg["rew"][idx])
                qf_adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(qf_losses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("CMAES Policy Optimization")
        # Make two q network equal
        assign_target_q_eq_eval_q()

        # CMAES
        assign_pi_zero_eq_new() #memorize the p0

        weights = pi.get_trainable_variables()
        layer_params = [v for v in weights if v.name.split("/")[1].startswith(
            "pol")]
        # if i + 1 < len(weights):
        #     layer_params = [weights[i], weights[i + 1]]
        # else:
        #     layer_params = [weights[i]]
        #     if len(layer_params) <= 1:
        #         layer_params = [weights[i - 1], weights[i]]
        layer_params_flat = get_layer_flat(layer_params)
        index, init_uniform_layer_weights = uniform_select(layer_params_flat,
                                                           1000)
        opt = cma.CMAOptions()
        opt['tolfun'] = max_fitness
        opt['popsize'] = popsize
        opt['maxiter'] = gensize
        opt['verb_disp'] = 0
        opt['verb_log'] = 0
        opt['seed'] = seed
        opt['AdaptSigma'] = True
        # opt['bounds'] = bounds
        sigma1 = sigma - 0.001 * iters_so_far
        if sigma1 < 0.0001:
            sigma1 = 0.0001
        print("Sigma=", sigma1)
        es = cma.CMAEvolutionStrategy(init_uniform_layer_weights,
                                      sigma1, opt)
        best_solution = init_uniform_layer_weights.astype(
            np.float64)
        best_fitness = np.inf
        costs = None
        while True:
            if es.countiter >= opt['maxiter']:
                break
            solutions = es.ask()
            assign_backup_eq_new() #backup current policy, after Q and V have been trained
            if KL_Condition:
                for id, solution in enumerate(solutions):
                    new_variable = set_uniform_weights(layer_params_flat, solution, index)
                    set_layer_flat(layer_params, new_variable)
                    i = 0
                    mean_kl_const = mean_Kl(ob)[0]
                    while(mean_kl_const > 0.5):
                        i+=1
                        # solutions[id] = es.ask(number = 1, xmean = np.take(pi_zero_get_flat(), index),
                        # sigma_fac = 0.9 ** i)
                        solutions[id] = es.ask(number = 1, sigma_fac = 0.9 ** i)[0]
                        new_variable = set_uniform_weights(layer_params_flat, solutions[id], index)
                        set_layer_flat(layer_params, new_variable)
                        mean_kl_const = mean_Kl(ob)[0]
                        logger.log("Regenerate Solution for " +str(i)+ " times for ID:" + str(id) + " mean_kl:" + str(mean_kl_const))

            if mean_action_Condition:
                for id, solution in enumerate(solutions):
                    new_variable = set_uniform_weights(layer_params_flat, solution, index)
                    set_layer_flat(layer_params, new_variable)
                    i = 0
                    # mean_act_dist = np.sqrt(np.dot(np.array(mean_pi_actions(ob)).flatten() - np.array(mean_pi_zero_actions(ob)).flatten(),
                    #                                np.array(mean_pi_actions(ob)).flatten() - np.array(mean_pi_zero_actions(ob)).flatten()))
                    abs_act_dist = np.mean(np.abs(np.array(mean_pi_actions(ob)).flatten() - np.array(mean_pi_zero_actions(ob)).flatten()))
                    while(abs_act_dist > 0.01):
                        i+=1
                        # solutions[id] = es.ask(number = 1, xmean = np.take(pi_zero_get_flat(), index),
                        # sigma_fac = 0.9 ** i)
                        solutions[id] = es.ask(number = 1, sigma_fac = 0.999 ** i)[0]
                        new_variable = set_uniform_weights(layer_params_flat, solutions[id], index)
                        set_layer_flat(layer_params, new_variable)
                        # mean_act_dist = np.sqrt(np.dot(np.array(mean_pi_actions(ob)).flatten() - np.array(mean_pi_zero_actions(ob)).flatten(),
                        #                            np.array(mean_pi_actions(ob)).flatten() - np.array(mean_pi_zero_actions(ob)).flatten()))
                        abs_act_dist = np.mean(np.abs(np.array(mean_pi_actions(ob)).flatten() - np.array(mean_pi_zero_actions(ob)).flatten()))
                        logger.log("Regenerate Solution for " +str(i)+ " times for ID:" + str(id) + " mean_action_dist:" + str(abs_act_dist))
            assign_new_eq_backup() # Restore the backup
            segs = []
            ob_segs = None
            costs = []
            lens = []
            # Evaluation

            a_func =get_A_estimation(ob,ob,mean_pi_actions(ob)[0])
            # a_func = (a_func - np.mean(a_func)) / np.std(a_func)
            a_func_pi_zero = get_A_pi_zero_estimation(ob,ob,mean_pi_zero_actions(ob)[0])
            print("A-pi-zero:", np.mean(a_func_pi_zero))
            print("A-pi-best:",
                  np.mean(a_func))
            print()
            for id, solution in enumerate(solutions):
                new_variable = set_uniform_weights(layer_params_flat, solution, index)
                set_layer_flat(layer_params, new_variable)
                new_a_func= get_A_estimation(ob,
                                          ob,
                                          np.array(mean_pi_actions(ob)).transpose().reshape((len(ob), 1)))
                # new_a_func = (new_a_func - np.mean(new_a_func)) / np.std(new_a_func)
                print("A-pi" + str(id + 1), ":", np.mean(new_a_func))
                coeff1 = 0.9
                coeff2 = 0.9
                cost = - (np.mean(new_a_func))
                # cost = - (np.mean(new_a_func) - coeff1*
                #           np.sqrt(np.dot(pi_get_flat() - pi_zero_get_flat(),
                #                          pi_get_flat() - pi_zero_get_flat()))
                #           - coeff2 * mean_Kl(ob)[0])
                # new_a_funcs =
                costs.append(cost)
                assign_new_eq_backup() # Restore the backup
            # l2_decay = compute_weight_decay(0.999, solutions).reshape((np.array(costs).shape))
            # costs += l2_decay
            # costs, real_costs = fitness_normalization(costs)
            print(costs)
            # costs, real_costs = fitness_rank(costs)
            # es.tell(solutions=solutions, function_values = costs)
            es.tell_real_seg(solutions = solutions, function_values = costs, real_f = costs, segs = None)
            # if -min(costs) >= np.mean(a_func):
            if min(costs) <= best_fitness:
                print("Update Policy by CMAES")
                # best_solution = np.copy(es.result[0])
                # best_fitness = -es.result[1]
                best_solution = solutions[np.argmin(costs)]
                best_fitness = min(costs)
                best_layer_params_flat = set_uniform_weights(layer_params_flat,
                                                             best_solution,
                                                             index)
                set_layer_flat(layer_params, best_layer_params_flat)
            # assign_pi_zero_eq_new()
            # if mean_Kl(ob)[0] > 0.05: # Check the kl diverge
            #     print("mean_kl:", mean_Kl(ob)[0])
            #     print("Cancel updating")
            #     assign_new_eq_backup()
            # else:
            #     assign_pi_zero_eq_new() #memorize the p0
            print("Generation:", es.countiter)
            print("Best Solution Fitness:", best_fitness)
            # set old parameter values to new parameter values
            # break
        # Record CMAES-Updated Pi behaviors 25 times
        eval_seg = eval_gen.__next__()
        test_rew_buffer.append(eval_seg["ep_rets"])
        assign_pi_zero_eq_new() #Update the p0

        print("Pi 0 Performance:", test_rew_buffer[0])
        print("Pi 1 Performance:", test_rew_buffer[1])

        print("Pi 0 Mean {0} Std {1}".format(np.mean(test_rew_buffer[0]), np.std(test_rew_buffer[0])))
        print("Pi 1 Mean {0} Std {1}".format(np.mean(test_rew_buffer[1]), np.std(test_rew_buffer[1])))
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
# def fitness_rank(x):
#     x = np.asarray(x).flatten()
#     ranks = np.empty(len(x))
#     ranks[x.argsort()] = np.arange(len(x))
#     ranks /= (len(x) - 1)
#     ranks -= .5
#     return ranks, x


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_layer_flat(var_list):
    op = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in var_list])
    return tf.get_default_session().run(op)

def set_layer_flat(old_var_list, var_list):
    dtype = tf.float32
    shapes = list(map(U.var_shape, old_var_list))
    total_size = np.sum([U.intprod(shape) for shape in shapes])

    theta = theta = tf.placeholder(dtype, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, old_var_list):
        size = U.intprod(shape)
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    return tf.get_default_session().run(op, feed_dict={theta: var_list})
