# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import sys
import random
import cPickle as pickle



class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 512
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = '../vgg_16_logs'
model_dir = '../../model/vgg_10k'
data_dir = '../../data/result/10K_data'


def prepare_data():
    print("======Loading data======")

    with open(data_dir, 'rb') as r:
        data_dict = pickle.load(r)

    org_img = data_dict['origin_img']
    adv_img = data_dict['adv_img']
    org_labels = data_dict['origin_labels']
    adv_labels = data_dict['adv_labels']

    test_data = np.array(adv_img.tolist() + org_img.tolist()).astype(float)
    test_labels = np.array(org_labels.tolist() + org_labels.tolist()).astype(float)


    print("======Prepare Finished======")

    print test_data.shape
    print test_labels.shape

    return test_data, test_labels




def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)




def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')




def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)




def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)




def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])


    if padding:
        oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return new_batch




def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch




def data_preprocessing(x_train,x_test):


    x_train = (np.asarray(x_train) / 255.0).astype('float32')
    x_test = (np.asarray(x_test) / 255.0).astype('float32')

    # print np.shape(x_train), np.shape(x_test)
    return x_train, x_test




def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch




def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001




def run_testing(sess):
    pre_index = 0
    add = 1000

    acc = 0
    for it in range(9):
        print "iter: %d" % it
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index = pre_index + add
        loss_, acc_  = sess.run([cross_entropy, accuracy],
                                feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
        print "loss: %.4f, acc: %.4f" % (loss_, acc_)
        if it < 8:
            acc += acc_*1000/8526
        else:
            acc += acc_*526/8526
    print "total acc : %.4f" % acc





if __name__ == '__main__':


    test_x, test_y = prepare_data()

    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32,[None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)


    # build_network
    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1) + b_conv1_1))


    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2))
    output = max_pool(output, 2, 2, "pool1")


    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1))


    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2))
    output = max_pool(output, 2, 2, "pool2")


    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    output = tf.nn.relu( batch_norm(conv2d(output,W_conv3_1) + b_conv3_1))


    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2))


    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    output = tf.nn.relu( batch_norm(conv2d(output, W_conv3_3) + b_conv3_3))
    output = max_pool(output, 2, 2, "pool3")


    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1))


    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2))


    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3))
    output = max_pool(output, 2, 2)


    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1))


    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2))


    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3))
    #output = max_pool(output, 2, 2)


    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 2*2*512])


    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1) )
    output = tf.nn.dropout(output, keep_prob)


    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
    output = tf.nn.dropout(output, keep_prob)


    W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([10])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))


    # output  = tf.reshape(output,[-1,10])


    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output), name="cross_entropy")
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).\
        minimize(cross_entropy + l2 * weight_decay)


    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


    # initial an saver to save model
    saver = tf.train.Saver()


    with tf.Session() as sess:

        # sess.run(tf.global_variables_initializer())
        print "load model"
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        print "load finished"

        run_testing(sess)
