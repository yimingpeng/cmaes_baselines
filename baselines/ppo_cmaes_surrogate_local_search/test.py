#!/usr/bin/env python

"""Description:
"""

__author__ = "Yiming Peng"
__copyright__ = "Copyright 2018, baselines"
__credits__ = ["Yiming Peng"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Yiming Peng"
__email__ = "yiming.peng@ecs.vuw.ac.nz"
__status__ = "Prototype"

import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U


w = tf.variable()
b = tf.variable()
tf.tan(w*x + b)
with tf.variable_scope("split"):
    raw_input = tf.placeholder(tf.float32, shape = [None, 16, 9])
    split_inputs = tf.map_fn(lambda x: tf.split(x, 16, axis = 0), raw_input, dtype=[tf.float32] * 16)

with tf.variable_scope("output", reuse = tf.AUTO_REUSE):
    # split_input = tf.placeholder(tf.float32, shape = [None, 9])
    i = tf.constant(0)
    ijk_0 = (tf.constant(0), tf.reduce_sum(split_inputs[0][0]))
    c = lambda i, p: i < 16
    b = lambda i, p: (i+1, tf.reduce_sum(split_inputs[i][0]))
    result = tf.while_loop(c, b, ijk_0)

    output = tf.layers.dense(split_input, 1, name='final',
                             kernel_initializer=tf.constant_initializer(1.0))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    var = np.array([[[1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [2., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 3., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    a = sess.run(result, {raw_input: var})
    print(a)
    # print(sess.run(output, {split_input: np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])}))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     #
#     print(sess.run(result, {raw_input:np.array([[[1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [2., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 3., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                 [1., 1., 1., 1., 1., 1., 1., 1., 1.]]])}))
