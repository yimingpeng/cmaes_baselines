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

with tf.variable_scope("split"):
    raw_input = tf.placeholder(tf.float32, shape = [None, 16, 9])
    split_inputs = tf.map_fn(lambda x: tf.split(x, 16, axis = 0), raw_input, dtype=[tf.float32] * 16)
with tf.variable_scope("output", reuse = tf.AUTO_REUSE):
    split_input = tf.placeholder(tf.float32, shape = [None, 9])
    output = tf.layers.dense(split_input, 1, name='final',
                             kernel_initializer=tf.constant_initializer(1.0))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    a = sess.run(split_inputs, {raw_input: np.array([[[1., 1., 1., 1., 1., 1., 1., 1., 1.],
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
            [1., 1., 1., 1., 1., 1., 1., 1., 1.]]])})
    print(a)
    print(a[0][0])
    # print(sess.run(output, {split_input: np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.]])}))
    for i in range(16):
        print(sess.run(output, {split_input: a[i][0]}))
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
