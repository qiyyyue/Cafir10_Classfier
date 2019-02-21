# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import sys
import random
import cPickle as pickle
import os
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
import matplotlib.pyplot as plt

model_dir = "../model/vgg_v2"
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
model_save_path = '../model/vgg_v2'
data_dir = '../data/cifar-10-batches-py'
save_dir = '../data/result'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict




def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels




def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels




def prepare_data():
    print("======Loading data======")
    # download_data()
    image_dim = image_size * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')


    print(meta)
    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)


    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")


    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")


    return train_data, train_labels, test_data, test_labels




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
                                        is_training=False, updates_collections=None)




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

if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    # print type(test_x), type(test_y)

    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.Variable(tf.zeros((1, 32, 32, 3)), name="train_x")
    y_ = tf.placeholder(tf.float32, [None, class_num], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    learning_rate = tf.placeholder(tf.float32, name="lr")
    train_flag = tf.placeholder(tf.bool, name="train_flag")

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

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2))
    output = max_pool(output, 2, 2, "pool2")

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_1) + b_conv3_1))

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2))

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_3) + b_conv3_3))
    output = max_pool(output, 2, 2, "pool3")

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_1) + b_conv4_1))

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_2) + b_conv4_2))

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv4_3) + b_conv4_3))
    output = max_pool(output, 2, 2)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_1) + b_conv5_1))

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_2) + b_conv5_2))

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    output = tf.nn.relu(batch_norm(conv2d(output, W_conv5_3) + b_conv5_3))
    # output = max_pool(output, 2, 2)

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 2 * 2 * 512])

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc1) + b_fc1))
    output = tf.nn.dropout(output, keep_prob)

    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc2) + b_fc2))
    output = tf.nn.dropout(output, keep_prob)

    W_fc3 = tf.get_variable('fc3', shape=[4096, 10], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([10])
    output = tf.nn.relu(batch_norm(tf.matmul(output, W_fc3) + b_fc3))

    probs = tf.nn.softmax(output)
    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output),
                                   name="cross_entropy")
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


    # print "start testing"
    with tf.Session() as sess:
        print "load graph"
        exclude = ['train_x']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        print "load finished"

        graph = tf.get_default_graph()

        x_img = tf.placeholder(tf.float32, (1, 32, 32, 3))
        x_hat = x
        assign_op = tf.assign(x_hat, x_img)

        adv_lr = tf.placeholder(tf.float32, (), name="adv_lr")
        # y_hat = tf.placeholder(tf.int32, (), name="y_hat")
        #
        # labels = tf.one_hot(y_hat, 10)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_)
        optim_step = tf.train.GradientDescentOptimizer(
            adv_lr).minimize(0-loss, var_list=[x_hat])

        epsilon = tf.placeholder(tf.float32, ())

        below = x_img - epsilon
        above = x_img + epsilon
        projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
        # projected = tf.clip_by_value(x_hat, below, above)
        with tf.control_dependencies([projected]):
            project_step = tf.assign(x_hat, projected)

        demo_epsilon = 2.0 / 255.0  # a really small perturbation
        demo_lr = 1e-1
        demo_steps = 200

        org_labels = []
        adv_labels = []
        org_img = []
        adv_img = []

        total_org_labels = []
        total_adv_labels = []
        total_org_img = []
        total_adv_img = []

        ss_time = time.time()

        count_succ = 0
        count_fail = 0


        for i in range(499, -1, -1):
            print "processing img %d:" % i

            print "org_loss:", sess.run(cross_entropy, feed_dict={x: [test_x[i]], y_: [test_y[i]], keep_prob: 1.0, train_flag: False})

            s_time = time.time()

            img = test_x[i]

            # demo_target = random.randint(0, 9)
            # while demo_target == np.argmax(test_y[i]):
            #     demo_target = random.randint(0, 9)

            # print "target", demo_target, np.argmax(test_y[i])

            sess.run(assign_op, feed_dict={x_img: [img]})
            # demo_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype="float64")

            # print "_y", test_y[i]
            # print "train adv"
            for j in range(demo_steps):
                # gradient descent step
                _, loss_value = sess.run(
                    [optim_step, loss],
                    feed_dict={adv_lr: demo_lr, y_: [test_y[i]], keep_prob: 1.0, train_flag: False})
                # project step
                sess.run(project_step, feed_dict={x_img: [img], epsilon: demo_epsilon, keep_prob: 1.0, train_flag: False})
                if (j + 1) % 10 == 0:
                    print('step %d, loss=%g' % (j + 1, loss_value))

            adv = x_hat.eval()[0]
            # print demo_target

            prob = sess.run(tf.argmax(output, 1), feed_dict={x: [adv], keep_prob: 1.0, train_flag: False})
            print "target: ", np.argmax(test_y[i]), prob[0]

            # print "prob", prob
            # max_target = np.argmax(prob)
            # print "max prob index", max_target

            # print "adv_label", type(adv_label), adv_label


            if loss_value > 1:
                count_succ += 1
                org_labels.append(np.argmax(test_y[i]))
                adv_labels.append(prob[0])
                org_img.append(test_x[i])
                adv_img.append(adv)

                # plt.imshow(adv)
                # plt.show()
                # plt.imshow(test_x[i])
                # plt.show()

            else:
                count_fail += 1

            e_time = time.time()
            print "time cost: %ds" % (e_time - s_time)
            print "--------------------------------"

            if i%500 == 0:
                print "500 count:"
                print "succ: %d, fail: %d" % (count_succ, count_fail)

                dump_data = {"origin_lables": np.array(org_labels), "adv_labels": np.array(adv_labels), "origin_img": np.array(org_img),
                             "adv_img": np.array(adv_img)}
                dump_dir = os.path.join(save_dir, "%d"%i)
                with open(dump_dir, "wb") as wb:
                    pickle.dump(dump_data, wb)

                total_org_labels += org_labels
                total_adv_labels += adv_labels
                total_org_img += org_img
                total_adv_img += adv_img

                org_labels = []
                adv_labels = []
                org_img = []
                adv_img = []



        ee_time = time.time()
        print "total time cost: %ds" % (ee_time - ss_time)

        dump_data = {"origin_lables": np.array(total_org_labels), "adv_labels": np.array(total_adv_labels), "origin_img": np.array(total_org_img),
                             "adv_img": np.array(total_adv_img)}
        with open("test_r_5k_data.pk", "wb") as wb:
            pickle.dump(dump_data, wb)

