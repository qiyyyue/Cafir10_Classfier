import random

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    for i in range(100):
        demo_target = random.randint(0, 9)
        print demo_target

    # target = 6
    #
    # label = tf.one_hot(target, 10)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print sess.run(label)