# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import sys

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from SSD.tool.config import Config
from SSD.tool.get_anchors import get_Anchors
from SSD.tool.read_Data import readData
import SSD.tool.ssd_loss as  ssd_loss
from datetime import datetime

Zero = tf.constant(0, dtype=tf.float32)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        # [(grad0, var0),(grad1, var1),...]
        average_grads.append(grad_and_var)
    return average_grads


def new_conv2d(net, channel, stride, name):
    net = slim.conv2d(net, channel / 2, [1, 1], scope='%s_1' % name)
    if stride == 2:
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
    net = slim.conv2d(net, channel, [3, 3], stride=stride, padding='VALID', scope='%s_2' % name)
    return net
    pass


def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma


class SSD():
    def __init__(self, config):
        self.config = config

        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.anchors = get_Anchors(config.img_size, config.s_min, config.s_max, config.num_anchors, config.map_size,
                                   config.stride_size, )

        self.anchors = tf.constant(self.anchors)
        print(config.weight_decay, self.anchors)

    def build_net(self, Iter):
        im, bboxes, nums = Iter.get_next()
        im.set_shape(tf.TensorShape([None, self.config.img_size, self.config.img_size, 3]))
        im = im - self.Mean
        im = im[..., ::-1]
        batch_m = tf.shape(im)[0]

        C = []
        with slim.arg_scope([slim.max_pool2d], padding='SAME'):
            with tf.variable_scope('vgg_16'):
                net = slim.repeat(im, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                C.append(('conv4_3', net))
                net4 = net
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
            var_pre = tf.global_variables()[1:]
            C.append(('conv7', net))
            net7 = net
            net = new_conv2d(net, 512, 2, 'conv8')
            net8 = net
            C.append(('conv8_2', net))
            net = new_conv2d(net, 256, 2, 'conv9')
            C.append(('conv9_2', net))
            net9 = net
            net = new_conv2d(net, 256, 1, 'conv10')
            net10 = net
            C.append(('conv10_2', net))
            net = new_conv2d(net, 256, 1, 'conv11')
            C.append(('conv11_2', net))
            net11 = net

            net4 = l2norm(net4, 20)

            with tf.variable_scope('detect'):
                with slim.arg_scope([slim.conv2d], activation_fn=None, ):
                    net4 = slim.conv2d(net4, self.config.num_anchors[0] * (self.config.num_cls + 1 + 4), [3, 3],
                                       scope='net4')
                    net7 = slim.conv2d(net7, self.config.num_anchors[1] * (self.config.num_cls + 1 + 4), [3, 3],
                                       scope='net7')
                    net8 = slim.conv2d(net8, self.config.num_anchors[2] * (self.config.num_cls + 1 + 4), [3, 3],
                                       scope='net8')
                    net9 = slim.conv2d(net9, self.config.num_anchors[3] * (self.config.num_cls + 1 + 4), [3, 3],
                                       scope='net9')
                    net10 = slim.conv2d(net10, self.config.num_anchors[4] * (self.config.num_cls + 1 + 4), [3, 3],
                                        scope='net10')
                    net11 = slim.conv2d(net11, self.config.num_anchors[5] * (self.config.num_cls + 1 + 4), [3, 3],
                                        scope='net11')
            net4 = tf.reshape(net4, (batch_m, -1, self.config.num_cls + 1 + 4), )
            net7 = tf.reshape(net7, (batch_m, -1, self.config.num_cls + 1 + 4), )
            net8 = tf.reshape(net8, (batch_m, -1, self.config.num_cls + 1 + 4), )
            net9 = tf.reshape(net9, (batch_m, -1, self.config.num_cls + 1 + 4), )
            net10 = tf.reshape(net10, (batch_m, -1, self.config.num_cls + 1 + 4), )
            net11 = tf.reshape(net11, (batch_m, -1, self.config.num_cls + 1 + 4), )

            net = tf.concat((net4, net7, net8, net9, net10, net11), axis=1, name='net')
            loss, Num = self.get_loss(net, bboxes, nums)
            return tf.reduce_sum(loss) / tf.reduce_sum(Num + 1e-10), var_pre

    def fn_map(self, x):
        net = x[0]
        bboxes = x[1][:x[2]]
        loss = tf.cond(tf.equal(x[2], 0), lambda: (Zero, Zero), lambda: ssd_loss.loss(net, self.anchors, bboxes))

        return loss

    def get_loss(self, net, bboxes, nums):
        loss, Num = tf.map_fn(self.fn_map, [net, bboxes, nums], (tf.float32, tf.float32))
        return tf.reduce_sum(loss), tf.reduce_sum(Num)

    def init(self, var_pre, sess):
        weights = np.load(self.config.pre_model, encoding='latin1')
        weights = weights.item()
        keys = weights.keys()
        keys = sorted(keys)
        c = 0
        for i in range(int(len(var_pre) / 2)):
            key = keys[i]
            b = weights[key]['biases']
            w = weights[key]['weights']
            pre_w = var_pre[i * 2]
            pre_b = var_pre[i * 2 + 1]
            print(pre_w.name, pre_b.name, key, c, c + 1)
            sess.run(pre_w.assign(w), )
            sess.run(pre_b.assign(b), )
            c += 2

    def train(self, ):
        shard_nums = self.config.gpus
        base_lr = self.config.lr
        print('*********************', shard_nums, base_lr, self.config.batch_size_per_GPU)
        steps = tf.Variable(0.0, name='ssd_steps', trainable=False)
        x = 2
        lr = tf.case({steps < 40000.0 * x: lambda: base_lr, steps < 50000.0 * x: lambda: base_lr / 10},
                     default=lambda: base_lr / 100)
        tower_grads = []
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        var_reuse = False
        Iter_list = []

        for i in range(shard_nums):
            with tf.device('/gpu:%d' % i):
                loss = 0
                Iter = readData(self.config.files, self.config, batch_size=self.config.batch_size_per_GPU,
                                num_threads=16,
                                shuffle_buffer=1024,
                                num_shards=shard_nums, shard_index=i)

                Iter_list.append(Iter)
                with tf.variable_scope('', reuse=var_reuse):
                    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                        weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                        if i == 0:
                            pre_loss, var_pre = self.build_net(Iter)
                        else:
                            pre_loss, _ = self.build_net(Iter)

                    var_reuse = True
                    loss += pre_loss
                train_vars = tf.trainable_variables()
                l2_loss = tf.losses.get_regularization_losses()
                l2_re_loss = tf.add_n(l2_loss)

                ssd_train_loss = pre_loss + l2_re_loss
                print('********', ssd_train_loss)
                grads_and_vars = opt.compute_gradients(ssd_train_loss, train_vars)
                tower_grads.append(grads_and_vars)
        # for v in tf.global_variables():
        #     print(v)
        grads = average_gradients(tower_grads)
        grads = list(zip(*grads))[0]
        grads, norm = tf.clip_by_global_norm(grads, 20.0)

        train_op = opt.apply_gradients(zip(grads, train_vars), global_step=steps)

        saver = tf.train.Saver(max_to_keep=200)
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            self.init(var_pre, sess)

            # saver.restore(sess, file)

            for Iter in Iter_list:
                sess.run(Iter.initializer)

            for i in range(00000, 60010 * x):
                if i % 20 == 0:
                    _, loss_, a, b, c, d = sess.run(
                        [train_op, ssd_train_loss, loss, l2_re_loss, norm, lr])
                    print(datetime.now(), 'ssd_loss:%.4f' % loss_, 'loss:%.4f' % a, 'l2_re_loss:%.4f' % b,
                          'norm:%.4f' % c, d, i)
                else:
                    sess.run(train_op)

                if (i + 1) % 5000 == 0 or ((i + 1) % 1000 == 0 and i < 10000):
                    saver.save(sess, os.path.join('./models/', 'SSD300_2x.ckpt'), global_step=i + 1)

            pass


if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']
    num_anchors = [4, 6, 6, 6, 4, 4]
    map_size = [38, 19, 10, 5, 3, 1]
    stride_size = [8, 16, 32, 64, 100, 300]

    pre_model = r'/home/zhai/PycharmProjects/Demo35/SSD/train/VGG_1024.npy'
    config = Config(True, Mean, files, pre_model, s_min=0.2, img_size=300, batch_size_per_GPU=16, gpus=2,
                    weight_decay=0.0005,
                    jitter_ratio=[0.3, 0.5, 0.7], crop_iou=0.45, keep_ratio=0.2, img_scale_size=[212, 150, 106, 75],
                    num_anchors=[4, 6, 6, 6, 4, 4],
                    stride_size=[8, 16, 32, 64, 100, 300], map_size=[38, 19, 10, 5, 3, 1])
    ssd = SSD(config)
    ssd.train()

    pass
