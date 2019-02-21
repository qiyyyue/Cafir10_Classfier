import random

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    for i in range(1000, 1, -1):
        print i

    # target = 6
    #
    # label = tf.one_hot(target, 10)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print sess.run(label)