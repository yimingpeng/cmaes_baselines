import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='small'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        x = ob / 255.0
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def get_Flat_variables(self):
        # weights = [v for v in self.get_trainable_variables()]
        return U.GetFlat(self.get_trainable_variables())

    def set_Flat_variables(self, var_list):
        set_from_flat = U.SetFromFlat(self.get_trainable_variables())
        set_from_flat(var_list)

    def get_Layer_Flat_variables(self, var_list):
        return U.GetFlat(var_list)

    def set_Layer_Flat_variables(self, old_var_list, var_list):
        set_from_flat = U.SetFromFlat(old_var_list)
        set_from_flat(var_list)
