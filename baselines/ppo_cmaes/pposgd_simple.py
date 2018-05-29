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
from baselines.common.mpi_moments import mpi_moments
from baselines.ppo_cmaes_per_layer.cnn_policy import CnnPolicy

test_rewbuffer = deque(maxlen = 100)  # test buffer for episode rewards


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
        ac, vpred = pi.act(stochastic, ob)
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


def traj_segment_generator(pi, env, horizon, stochastic):
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
    next_obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
            result_record(best_fitness)
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


def traj_segment_generator_cmaes(pi, env, horizon, stochastic, eval_iters):
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
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    ep_num = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if (t > 0 and t % horizon == 0) or ep_num >= eval_iters:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
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
        vpreds[i] = vpred
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


def fitness_evaluation(atarg):
    a_estimate = np.mean(atarg)
    return a_estimate


def result_record(best_fitness_so_far):
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
    logger.log("********** Iteration %i ************" % iters_so_far)
    # rewbuffer.append(best_fitness_so_far)
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


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis = 1)


def uniform_select(weight, num_of_weights):
    length = len(weight) if len(weight) < num_of_weights else \
        num_of_weights
    index = np.random.choice(range(len(
        weight)), length, replace = False)
    return index, np.take(weight, index)


def set_uniform_weights(original_weight, new_weight, index):
    result_weight = np.copy(original_weight)
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
    evalpi = policy_fn("evalpi", ob_space, ac_space)  # Network for old policy
    # used for CMAES evaluation
    atarg = tf.placeholder(dtype = tf.float32, shape = [
        None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype = tf.float32, shape = [None])  # Empirical return

    reward = tf.placeholder(dtype = tf.float32, shape = [None]) #step rewards
    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name = "ob")
    next_ob = U.get_placeholder_cached(name = "next_ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
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

    y = reward + gamma * tf.squeeze(pi.vpred)
    qf_loss = tf.reduce_mean(tf.square(y - pi.qpred))
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    qf_losses = [qf_loss]
    vf_losses = [vf_loss]
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
        # Policy + Value function, the final layer
        var_list = lin_var_list + vf_var_list + pol_var_list
        print(var_list)
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
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    qf_lossandgrad = U.function([ob, ac, next_ob, lrmult, reward],
                                qf_losses + [U.flatgrad(qf_loss, qf_var_list)])
    vf_lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
                             losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon = adam_epsilon)

    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)

    qf_adam = MpiAdam(qf_var_list, epsilon = adam_epsilon)

    assign_old_eq_new = U.function([], [], updates = [tf.assign(oldv, newv)
                                                      for (oldv, newv) in zipsame(
            oldpi.get_variables(), pi.get_variables())])

    assign_eval_eq_new = U.function([], [], updates = [tf.assign(evalv, newv)
                                                      for (evalv, newv) in zipsame(
            evalpi.get_variables(), pi.get_variables())])

    assign_new_eq_eval= U.function([], [], updates = [tf.assign(newv,evalv)
                                                      for (newv,evalv) in zipsame(
            pi.get_variables(), evalpi.get_variables())])

    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    get_prob = U.function([ob, ac], [tf.exp(pi.pd.logp(ac))])

    get_A_estimation = U.function([ob, next_ob, ac], [pi.qpred - pi.vpred])
    get_eval_A_estimation = U.function([ob, next_ob, ac], [evalpi.qpred - evalpi.vpred])

    mean_actions = U.function([ob], [pi.pd.mode()])
    mean_eval_actions = U.function([ob], [evalpi.pd.mode()])

    U.initialize()
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
    # Prepare for rollouts
    # ----------------------------------------
    # assign pi to eval_pi
    # Build generator for all solutions
    actors = []
    best_fitness = 0
    for i in range(popsize):
        newActor = traj_segment_generator_cmaes(pi, env,
                                                timesteps_per_actorbatch,
                                                stochastic = True,
                                                eval_iters = eval_iters)
        actors.append(newActor)
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic = True)

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

        # Use atarg the advantage function estimation to do evaluation

        # Use CMAES layer-wised train
        # best_fitness = (np.mean(rews) + best_fitness)/2
        # CMAES

        assign_eval_eq_new()
        weights = evalpi.get_trainable_variables()
        pi_weights = pi.get_trainable_variables()
        if i >= len(weights):
            i = 0
        while i < len(weights):
            # Consider both q-function and v-function
            if weights[i].name.split("/")[1] == "vf":
                i += 1
                continue
            print("Layer: ", i, '+', i + 1)
            print("Layer-Name", weights[i].name)
            if i + 1 < len(weights):
                layer_params = [weights[i], weights[i + 1]]
                pi_layer_params = [pi_weights[i], pi_weights[i + 1]]
            else:
                layer_params = [weights[i]]
                pi_layer_params = [pi_weights[i]]
                if len(layer_params) <= 1:
                    layer_params = [weights[i - 1], weights[i]]
            layer_params_flat = evalpi.get_Layer_Flat_variables(layer_params)()
            index, init_uniform_layer_weights = uniform_select(layer_params_flat,
                                                               500)
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
            best_solution = np.copy(init_uniform_layer_weights.astype(
                np.float64))
            costs = None
            while True:
                if es.countiter >= opt['maxiter']:
                    break

                #Every generation re-esitmate advantage functions
                seg = seg_gen.__next__()
                add_vtarg_and_adv(seg, gamma, lam)

                # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                ob, next_ob, ac, atarg, tdlamret, reward = seg["ob"], seg["next_ob"],  seg["ac"], seg["adv"], seg[
                    "tdlamret"], seg["rew"]
                vpredbefore = seg["vpred"]  # predicted value function before udpate
                atarg = (
                                atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
                d = Dataset(dict(ob = ob, ac = ac, atarg = atarg, vtarg = tdlamret),
                            shuffle = not pi.recurrent)
                optim_batchsize = optim_batchsize or ob.shape[0]

                lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
                listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
                lens, rews = map(flatten_lists, zip(*listoflrpairs))
                lenbuffer.extend(lens)
                rewbuffer.extend(rews)

                #Re-train V function
                for _ in range(optim_epochs):
                    losses = []  # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        *vf_losses, g = vf_lossandgrad(batch["ob"], batch["ac"],
                                                       batch["atarg"], batch["vtarg"],
                                                       cur_lrmult)
                        vf_adam.update(g, optim_stepsize * cur_lrmult)
                    # logger.log(fmt_row(13, np.mean(vf_losses, axis=0)))

                # Random select tansitions to train Q
                random_idx = []
                len_repo = len(seg["ob"])
                optim_epochs_q = int(len_repo/optim_batchsize)
                for _ in range(optim_epochs_q):
                    random_idx.append(np.random.choice(range(len_repo), optim_batchsize))

                #Re-train q function
                for _ in range(optim_epochs_q):
                    losses = []  # list of tuples, each of which gives the loss for a minibatch
                    for idx in random_idx:
                        *qf_losses, g = qf_lossandgrad(seg["next_ob"][idx], seg["ac"][idx], seg["ob"][idx],
                                                       cur_lrmult, seg["rew"][idx])
                        qf_adam.update(g, optim_stepsize * cur_lrmult)
                    # logger.log(fmt_row(13, np.mean(vf_losses, axis=0)))

                assign_eval_eq_new()
                solutions = es.ask()
                segs = []
                ob_segs = None
                costs = []
                lens = []
                # Evaluation
                # for id, solution in enumerate(solutions):
                #     new_variable = set_uniform_weights(layer_params_flat, solution, index)
                #     pi.set_Layer_Flat_variables(layer_params, new_variable)
                #     indv_seg = actors[id].__next__()
                #     costs.append(-np.mean(indv_seg["ep_rets"]))
                #     lens.append(np.sum(indv_seg["ep_lens"]))
                #     segs.append(indv_seg)
                #     if ob_segs is None:
                #         ob_segs = {'ob': np.copy(indv_seg['ob'])}
                #     else:
                #         ob_segs['ob'] = np.append(ob_segs['ob'], indv_seg['ob'], axis=0)
                layer_params_flat = evalpi.get_Layer_Flat_variables(layer_params)()
                original_layer_weights = np.take(layer_params_flat, index)

                for id, solution in enumerate(solutions):

                    print("A-pi0:",
                          fitness_evaluation(get_A_estimation(ob, ob, np.array(mean_actions(ob)).transpose().reshape((len(ob), )))))
                    new_variable = set_uniform_weights(layer_params_flat, solution, index)
                    evalpi.set_Layer_Flat_variables(layer_params, new_variable)
                    # indv_seg = actors[id].__next__()
                    # costs.append(-np.mean(indv_seg["ep_rets"]))
                    new_a_estimation = fitness_evaluation(
                        get_eval_A_estimation(ob, ob, np.array(mean_eval_actions(ob)).transpose().reshape((len(ob), ))))
                    print("pi",id, ":", new_a_estimation)
                    costs.append(-new_a_estimation)

                    evalpi.set_Layer_Flat_variables(layer_params, original_layer_weights)
                    # lens.append(np.sum(indv_seg["ep_lens"]))
                    # segs.append(indv_seg)
                    # if ob_segs is None:
                    #     ob_segs = {'ob': np.copy(indv_seg['ob'])}
                    # else:
                    #     ob_segs['ob'] = np.append(ob_segs['ob'], indv_seg['ob'], axis=0)
                # if (np.array(costs) < 0).all():
                #     print("all fitness < 0, regenerate solutions")
                #     continue
                # l2_decay = compute_weight_decay(0.999, solutions).reshape((np.array(costs).shape))
                # costs += l2_decay
                # costs, real_costs = fitness_normalization(costs)
                print(costs)
                costs, real_costs = fitness_rank(costs)
                # es.tell(solutions=solutions, function_values = costs)
                es.tell_real_seg(solutions = solutions, function_values = costs, real_f = costs, segs = None)
                # if -es.result[1] >= best_fitness:
                print("Update Policy by CMAES")
                best_solution = np.copy(es.result[0])
                best_fitness = -es.result[1]
                best_layer_params_flat = set_uniform_weights(layer_params_flat,
                                                             best_solution,
                                                             index)
                evalpi.set_Layer_Flat_variables(layer_params, best_layer_params_flat)
                print("Generation:", es.countiter)
                print("Best Solution Fitness:", best_fitness)

                assign_new_eq_eval()
                assign_old_eq_new()  # set old parameter values to new parameter values
                # if hasattr(pi, "ob_rms"): pi.ob_rms.update(
                #     ob_segs['ob'])  # update running mean/std for policy
            i += 2
            # break
        #Reestimate Advantage function based on the newly updated Pi
        # seg = seg_gen.__next__()
        # add_vtarg_and_adv(seg, gamma, lam)
        #
        # # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        # ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg[
        #     "tdlamret"]
        # vpredbefore = seg["vpred"]  # predicted value function before udpate
        # atarg = (
        #                 atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        # d = Dataset(dict(ob = ob, ac = ac, atarg = atarg, vtarg = tdlamret),
        #                     shuffle = not pi.recurrent)
        # optim_batchsize = optim_batchsize or ob.shape[0]

        # PPO training
        # assign_old_eq_new()  # set old parameter values to new parameter values
        # logger.log("Optimizing...")
        # logger.log(fmt_row(13, loss_names))
        # # Optimize the value function to keep it up.
        # for _ in range(optim_epochs):
        #     losses = []  # list of tuples, each of which gives the loss for a minibatch
        #     for batch in d.iterate_once(optim_batchsize):
        #         *vf_losses, g = vf_lossandgrad(batch["ob"], batch["ac"],
        #                                        batch["atarg"], batch["vtarg"],
        #                                        cur_lrmult)
        #         vf_adam.update(g, optim_stepsize * cur_lrmult)
            # logger.log(fmt_row(13, np.mean(vf_losses, axis=0)))

        # Here we do a bunch of optimization epochs over the data
        # for _ in range(optim_epochs):
        #     losses = []  # list of tuples, each of which gives the loss for a minibatch
        #     for batch in d.iterate_once(optim_batchsize):
        #         *newlosses, g = lossandgrad(batch["ob"], batch["ac"],
        #                                     batch["atarg"], batch["vtarg"],
        #                                     cur_lrmult)
        #         adam.update(g, optim_stepsize * cur_lrmult)
        #         losses.append(newlosses)
        #     logger.log(fmt_row(13, np.mean(losses, axis = 0)))

        # logger.log("Evaluating losses...")
        # losses = []
        # for batch in d.iterate_once(optim_batchsize):
        #     newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"],
        #                                batch["vtarg"], cur_lrmult)
        #     losses.append(newlosses)
        # meanlosses, _, _ = mpi_moments(losses, axis = 0)
        iters_so_far += 1
        episodes_so_far += sum(lens)


def fitness_normalization(x):
    x = np.asarray(x).flatten()
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, x

def fitness_rank(x):
    x = np.asarray(x).flatten()
    ranks = np.empty(len(x))
    ranks[x.argsort()] = np.arange(len(x))
    ranks /= (len(x) - 1)
    ranks -= .5
    return ranks, x

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
