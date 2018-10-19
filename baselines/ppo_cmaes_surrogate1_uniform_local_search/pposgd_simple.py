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
    ep_num = 0

    while True:
        ac, vpred, a_prop = pi.act(stochastic, ob)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)
        # ac = np.clip(ac, env.action_space.low, env.action_space.high)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0 and ep_num >= 5:
            yield {"ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_num = 0
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

        ob, rew, new, _ = env.step(ac)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_num += 1
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def traj_segment_generator(pi, env, horizon, stochastic, eval_seq):
    # Trajectories generators
    global timesteps_so_far, rewbuffer, lenbuffer
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
    act_props = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    traj_index = []
    index_count = 0
    record = False
    while True:
        if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
            record = True
            # result_record()
        prevac = ac
        ac, vpred, act_prop = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            if record:
                ob = env.reset()
                eval_seg = eval_seq.__next__()
                rewbuffer.extend(eval_seg["ep_rets"])
                lenbuffer.extend(eval_seg["ep_lens"])
                result_record()
                record = False
            yield {"ob": obs, "next_ob": next_obs, "rew": rews, "vpred": vpreds, "act_props": act_props, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "traj_index": traj_index}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            index_count = 0
            traj_index = []
            cur_ep_ret = 0
            cur_ep_len = 0

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        act_props[i] = act_prop
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ac = np.clip(ac, env.action_space.low, env.action_space.high)
        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        next_obs[i] = ob

        cur_ep_ret += rew
        cur_ep_len += 1
        timesteps_so_far += 1
        if new:
            if record:
                ob = env.reset()
                eval_seg = eval_seq.__next__()
                rewbuffer.extend(eval_seg["ep_rets"])
                lenbuffer.extend(eval_seg["ep_lens"])
                result_record()
                record = False
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            traj_index.append(index_count)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1
        index_count += 1


def result_record():
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart, best_fitness
    print(rewbuffer)
    # if best_fitness != -np.inf:
    #     rewbuffer.append(best_fitness)
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


def uniform_select(weights, proportion):
    num_of_weights = int(proportion * len(weights))
    assert num_of_weights != 0  # make sure there are something to be selected
    length = len(weights) if len(weights) < num_of_weights else \
        num_of_weights
    index = np.random.choice(range(len(
        weights)), length, replace = False)
    return index, np.take(weights, index, axis = 0)

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
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

    pi_params = tf.placeholder(dtype = tf.float32, shape = [None])
    old_pi_params = tf.placeholder(dtype = tf.float32, shape = [None])
    atarg = tf.placeholder(dtype = tf.float32, shape = [None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype = tf.float32, shape = [None])  # Empirical return

    lrmult = tf.placeholder(name = 'lrmult', dtype = tf.float32,
                            shape = [])  # learning rate multiplier, updated with schedule

    layer_clip = tf.placeholder(name = 'layer_clip', dtype = tf.float32,
                                shape = [])  # learning rate multiplier, updated with schedule

    bound_coeff = tf.placeholder(name = 'bound_coeff', dtype = tf.float32,
                                 shape = [])  # learning rate multiplier, updated with schedule

    clip_param = clip_param * lrmult * layer_clip  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name = "ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - (oldpi.pd.logp(ac)+1e-8))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    vf_losses = [vf_loss]
    vf_loss_names = ["vf_loss"]

    pol_loss = pol_surr + pol_entpen
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "vf")]
    pol_var_list = [v for v in var_list if v.name.split("/")[1].startswith(
        "pol")]

    layer_var_list = []
    for i in range(pi.num_hid_layers):
        layer_var_list.append([v for v in pol_var_list if v.name.split("/")[2].startswith(
            'fc%i' % (i + 1))])
    logstd_var_list = [v for v in pol_var_list if v.name.split("/")[2].startswith(
        "logstd")]
    if len(logstd_var_list) != 0:
        layer_var_list.append([v for v in pol_var_list if v.name.split("/")[2].startswith(
            "final")] + logstd_var_list)

    vf_lossandgrad = U.function([ob, ac, ret, lrmult],
                                vf_losses + [U.flatgrad(vf_loss, vf_var_list)])

    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, layer_clip],
                             losses + [U.flatgrad(total_loss, var_list)])

    vf_adam = MpiAdam(vf_var_list, epsilon = adam_epsilon)
    adam = MpiAdam(var_list, epsilon = adam_epsilon)

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

    compute_pol_losses = U.function([ob, ac, atarg, ret, lrmult, layer_clip],
                                    [pol_loss, pol_surr, pol_entpen, meankl])

    compute_v_pred = U.function([ob], [pi.vpred])

    a_prob = tf.exp(pi.pd.logp(ac))
    compute_a_prob = U.function([ob, ac], [a_prob])

    U.initialize()

    layer_set_operate_list = []
    layer_get_operate_list = []
    for var in layer_var_list:
        set_pi_layer_flat_params = U.SetFromFlat(var)
        layer_set_operate_list.append(set_pi_layer_flat_params)
        get_pi_layer_flat_params = U.GetFlat(var)
        layer_get_operate_list.append(get_pi_layer_flat_params)

    # get_pi_layer_flat_params = U.GetFlat(pol_var_list)
    # set_pi_layer_flat_params = U.SetFromFlat(pol_var_list)

    vf_adam.sync()

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

    best_fitness = -np.inf

    eval_seq = traj_segment_generator_eval(pi, env,
                                           timesteps_per_actorbatch,
                                           stochastic = False)
    # eval_gen = traj_segment_generator_eval(pi, test_env, timesteps_per_actorbatch, stochastic = True)  # For evaluation
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic = True, eval_seq=eval_seq)  # For train V Func

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    indices = []  # maintain all selected indices for each iteration

    opt = cma.CMAOptions()
    opt['tolfun'] = max_fitness
    opt['popsize'] = popsize
    opt['maxiter'] = gensize
    opt['verb_disp'] = 0
    opt['verb_log'] = 0
    # opt['seed'] = seed
    opt['AdaptSigma'] = True
    # opt['bounds'] = bounds
    # opt['tolstagnation'] = 20
    ess = []
    seg = None
    segs = None
    sum_vpred = []
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
            cur_lrmult = max(1.0 - float(timesteps_so_far) / (max_timesteps), 0)
        else:
            raise NotImplementedError

        # epsilon = max(0.5 - float(timesteps_so_far) / (max_timesteps), 0) * cur_lrmult
        epsilon = max(0.5 * cur_lrmult, 0)
        # epsilon = 0.2
        sigma_adapted = max(sigma * cur_lrmult, 1e-8)
        # sigma_adapted = max(max(sigma - float(timesteps_so_far) / (5000 * max_timesteps), 0) * cur_lrmult, 1e-8)
        # cmean_adapted = max(1.0 - float(timesteps_so_far) / (max_timesteps), 1e-8)
        # cmean_adapted = max(0.8 - float(time˚steps_so_far) / (2*max_timesteps), 1e-8)
        # if timesteps_so_far % max_timesteps == 10:
        # max_v_train_iter = int(max(max_v_train_iter * (1 - timesteps_so_far/(0.5*max_timesteps)), 1))
        logger.log("********** Iteration %i ************" % iters_so_far)
        if iters_so_far == 0:
            eval_seg = eval_seq.__next__()
            rewbuffer.extend(eval_seg["ep_rets"])
            lenbuffer.extend(eval_seg["ep_lens"])
            result_record()

        # Repository Train
        train_segs = {}
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(seg["ob"])  # update running mean/std for normalization

        # rewbuffer.extend(seg["ep_rets"])
        # lenbuffer.extend(seg["ep_lens"])
        #
        # if iters_so_far == 0:
        #     result_record()

        assign_old_eq_new()  # set old parameter values to new parameter values
        if segs is None:
            segs = seg
            segs["v_target"] = np.zeros(len(seg["ob"]), 'float32')
        elif len(segs["ob"]) >= 50000:
            segs["ob"] = np.take(segs["ob"], np.arange(timesteps_per_actorbatch, len(segs["ob"])), axis = 0)
            segs["next_ob"] = np.take(segs["next_ob"], np.arange(timesteps_per_actorbatch, len(segs["next_ob"])), axis = 0)
            segs["ac"] = np.take(segs["ac"], np.arange(timesteps_per_actorbatch, len(segs["ac"])), axis = 0)
            segs["rew"] = np.take(segs["rew"], np.arange(timesteps_per_actorbatch, len(segs["rew"])), axis = 0)
            segs["vpred"] = np.take(segs["vpred"], np.arange(timesteps_per_actorbatch, len(segs["vpred"])), axis = 0)
            segs["act_props"] = np.take(segs["act_props"], np.arange(timesteps_per_actorbatch, len(segs["act_props"])), axis = 0)
            segs["new"] = np.take(segs["new"], np.arange(timesteps_per_actorbatch, len(segs["new"])), axis = 0)
            segs["adv"] = np.take(segs["adv"], np.arange(timesteps_per_actorbatch, len(segs["adv"])), axis = 0)
            segs["tdlamret"] = np.take(segs["tdlamret"], np.arange(timesteps_per_actorbatch, len(segs["tdlamret"])), axis = 0)
            segs["ep_rets"] = np.take(segs["ep_rets"], np.arange(timesteps_per_actorbatch, len(segs["ep_rets"])), axis = 0)
            segs["ep_lens"] = np.take(segs["ep_lens"], np.arange(timesteps_per_actorbatch, len(segs["ep_lens"])), axis = 0)
            segs["v_target"] = np.take(segs["v_target"], np.arange(timesteps_per_actorbatch, len(segs["v_target"])), axis = 0)
            segs["ob"] = np.append(segs['ob'], seg['ob'], axis = 0)
            segs["next_ob"] = np.append(segs['next_ob'], seg['next_ob'], axis = 0)
            segs["ac"] = np.append(segs['ac'], seg['ac'], axis = 0)
            segs["rew"] = np.append(segs['rew'], seg['rew'], axis = 0)
            segs["vpred"] = np.append(segs['vpred'], seg['vpred'], axis = 0)
            segs["act_props"] = np.append(segs['act_props'], seg['act_props'], axis = 0)
            segs["new"] = np.append(segs['new'], seg['new'], axis = 0)
            segs["adv"] = np.append(segs['adv'], seg['adv'], axis = 0)
            segs["tdlamret"] = np.append(segs['tdlamret'], seg['tdlamret'], axis = 0)
            segs["ep_rets"] = np.append(segs['ep_rets'], seg['ep_rets'], axis = 0)
            segs["ep_lens"] = np.append(segs['ep_lens'], seg['ep_lens'], axis = 0)
            segs["v_target"] = np.append(segs['v_target'], np.zeros(len(seg["ob"]), 'float32'), axis = 0)
        else:
            segs["ob"] = np.append(segs['ob'], seg['ob'], axis = 0)
            segs["next_ob"] = np.append(segs['next_ob'], seg['next_ob'], axis = 0)
            segs["ac"] = np.append(segs['ac'], seg['ac'], axis = 0)
            segs["rew"] = np.append(segs['rew'], seg['rew'], axis = 0)
            segs["vpred"] = np.append(segs['vpred'], seg['vpred'], axis = 0)
            segs["act_props"] = np.append(segs['act_props'], seg['act_props'], axis = 0)
            segs["new"] = np.append(segs['new'], seg['new'], axis = 0)
            segs["adv"] = np.append(segs['adv'], seg['adv'], axis = 0)
            segs["tdlamret"] = np.append(segs['tdlamret'], seg['tdlamret'], axis = 0)
            segs["ep_rets"] = np.append(segs['ep_rets'], seg['ep_rets'], axis = 0)
            segs["ep_lens"] = np.append(segs['ep_lens'], seg['ep_lens'], axis = 0)
            segs["v_target"] = np.append(segs['v_target'], np.zeros(len(seg["ob"]), 'float32'), axis = 0)

        if iters_so_far == 0:
            ob, ac, tdlamret = seg["ob"], seg["ac"], seg["tdlamret"]
            d = Dataset(dict(ob = ob, ac = ac, vtarg = tdlamret), shuffle = not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            # Train V function
            # logger.log("Catchup Training V Func and Evaluating V Func Losses")
            for _ in range(max_v_train_iter):
                for batch in d.iterate_once(optim_batchsize):
                    *vf_loss, g = vf_lossandgrad(batch["ob"], batch["ac"], batch["vtarg"],
                                                   cur_lrmult)
                    vf_adam.update(g, optim_stepsize * cur_lrmult)
                # logger.log(fmt_row(13, np.mean(vf_losses, axis = 0)))
        else:

            # Update v target
            new = segs["new"]
            rew = segs["rew"]
            act_prob = np.asarray(compute_a_prob(segs["ob"], segs["ac"])).T
            importance_ratio = np.squeeze(act_prob)/(segs["act_props"] + np.ones(segs["act_props"].shape)*1e-8)
            segs["v_target"] = importance_ratio * (1/np.sum(importance_ratio)) * \
                               np.squeeze(rew + np.invert(new).astype(np.float32) * gamma * compute_v_pred(segs["next_ob"]))
            # train_segs["v_target"] = rew + np.invert(new).astype(np.float32) * gamma * compute_v_pred(train_segs["next_ob"])
            if len(segs["ob"]) >= 20000:
                train_times = int(max_v_train_iter / 2) if int(max_v_train_iter / 2) > 0 else 1
            else:
                train_times = 2
            for i in range(train_times):
                selected_train_index = np.random.choice(range(len(segs["ob"])), timesteps_per_actorbatch, replace = False)
                train_segs["ob"] = np.take(segs["ob"], selected_train_index, axis = 0)
                train_segs["next_ob"] = np.take(segs["next_ob"], selected_train_index, axis = 0)
                train_segs["ac"] = np.take(segs["ac"], selected_train_index, axis = 0)
                train_segs["rew"] = np.take(segs["rew"], selected_train_index, axis = 0)
                train_segs["vpred"] = np.take(segs["vpred"], selected_train_index, axis = 0)
                train_segs["new"] = np.take(segs["new"], selected_train_index, axis = 0)
                train_segs["adv"] = np.take(segs["adv"], selected_train_index, axis = 0)
                train_segs["tdlamret"] = np.take(segs["tdlamret"], selected_train_index, axis = 0)
                train_segs["v_target"] = np.take(segs["v_target"], selected_train_index, axis = 0)
                #
                ob, ac, v_target = train_segs["ob"], train_segs["ac"], train_segs["v_target"]
                d = Dataset(dict(ob = ob, ac = ac, vtarg = v_target), shuffle = not pi.recurrent)
                optim_batchsize = optim_batchsize or ob.shape[0]

                # Train V function
                # logger.log("Training V Func and Evaluating V Func Losses")
                # Train V function
                # logger.log("Catchup Training V Func and Evaluating V Func Losses")
                # logger.log("Train V - "+str(_))
                for _ in range(max_v_train_iter):
                    for batch in d.iterate_once(optim_batchsize):
                        *vf_loss, g = vf_lossandgrad(batch["ob"], batch["ac"], batch["vtarg"],
                                                       cur_lrmult)
                        vf_adam.update(g, optim_stepsize * cur_lrmult)
                    # logger.log(fmt_row(13, np.mean(vf_losses, axis = 0)))
                # seg['vpred'] = np.asarray(compute_v_pred(seg["ob"])).reshape(seg['vpred'].shape)
                # seg['nextvpred'] = seg['vpred'][-1] * (1 - seg["new"][-1])
                # add_vtarg_and_adv(seg, gamma, lam)


            ob, ac, atarg, v_target = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob = ob, ac = ac, atarg= atarg, vtarg = v_target), shuffle = not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]
            # Local search
            for _ in range(optim_epochs):
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                                1)
                    adam.update(g, optim_stepsize * cur_lrmult)

            # seg['vpred'] = np.asarray(compute_v_pred(seg["ob"])).reshape(seg['vpred'].shape)
            # seg['nextvpred'] = seg['vpred'][-1] * (1 - seg["new"][-1])
            # add_vtarg_and_adv(seg, gamma, lam)


        ob_po, ac_po, atarg_po, tdlamret_po = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        atarg_po = (atarg_po - atarg_po.mean()) / atarg_po.std()  # standardized advantage function estimate

        # opt['CMA_cmean'] = cmean_adapted
        # assign_old_eq_new()  # set old parameter values to new parameter values
        for i in range(len(layer_var_list)):
            # CMAES Train Policy
            assign_backup_eq_new()  # backup current policy
            flatten_weights = layer_get_operate_list[i]()

            if len(indices) < len(layer_var_list):
                selected_index, init_weights = uniform_select(flatten_weights,
                                                              0.5)  # 0.5 means 50% proportion of params are selected
                indices.append(selected_index)
            else:
                rand = np.random.uniform()
                # print("Random-Number:", rand)
                # print("Epsilon:", epsilon)
                if rand < epsilon:
                    selected_index, init_weights = uniform_select(flatten_weights, 0.5)
                    indices.append(selected_index)
                    # logger.log("Random: select new weights")
                else:
                    selected_index = indices[i]
                    init_weights = np.take(flatten_weights, selected_index)
            es = cma.CMAEvolutionStrategy(init_weights,
                                          sigma_adapted, opt)
            while True:
                if es.countiter >= gensize:
                    # logger.log("Max generations for current layer")
                    break
                # logger.log("Iteration:" + str(iters_so_far) + " - sub-train Generation for Policy:" + str(es.countiter))
                # logger.log("Sigma=" + str(es.sigma))
                # solutions = es.ask(sigma_fac = max(cur_lrmult, 1e-8))
                solutions = es.ask()
                # solutions = [np.clip(solution, -5.0, 5.0).tolist() for solution in solutions]
                costs = []
                lens = []

                assign_backup_eq_new()  # backup current policy

                for id, solution in enumerate(solutions):
                    np.put(flatten_weights, selected_index, solution)
                    layer_set_operate_list[i](flatten_weights)
                    # cost = compute_pol_losses(ob_po, ac_po, atarg_po, tdlamret_po, cur_lrmult, 1 / 4 * (i + 1))
                    cost = compute_pol_losses(ob_po, ac_po, atarg_po, tdlamret_po, cur_lrmult, 1.0)
                    costs.append(cost[0])
                    assign_new_eq_backup()
                # Weights decay
                l2_decay = compute_weight_decay(0.01, solutions)
                costs += l2_decay
                costs, real_costs = fitness_rank(costs)
                # logger.log("real_costs:"+str(real_costs))
                # best_solution = np.copy(es.result[0])
                # best_fitness = -es.result[1]
                es.tell_real_seg(solutions = solutions, function_values = costs, real_f = real_costs, segs = None)
                # best_solution = np.copy(solutions[np.argmin(costs)])
                # best_fitness = -real_costs[np.argmin(costs)]
                best_solution = es.result[0]
                best_fitness = es.result[1]
                np.put(flatten_weights, selected_index, best_solution)
                layer_set_operate_list[i](flatten_weights)
                # logger.log("Update the layer")
                # best_solution = es.result[0]
                # best_fitness = es.result[1]
                # logger.log("Best Solution Fitness:" + str(best_fitness))
                # set_pi_flat_params(best_solution)
            import gc
            gc.collect()

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
