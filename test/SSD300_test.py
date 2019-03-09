# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from SSD.tool.config import Config
from SSD.tool.get_anchors import get_Anchors
from SSD.tool.ssd_predict import predict
from datetime import datetime
from sklearn.externals import joblib




def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = tf.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = tf.concat((yx1, yx2), axis=-1)
    return bboxes


def new_conv2d(net, channel, stride, name):
    net = slim.conv2d(net, channel / 2, [1, 1], scope='%s_1' % name)
    if stride == 2:
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
    net = slim.conv2d(net, channel, [3, 3], stride=stride, padding='VALID', scope='%s_2' % name)
    return net
    pass


class SSD():
    def __init__(self, config):
        self.config = config
        print(self.config.Mean)
        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.anchors = get_Anchors(config.img_size, config.s_min, config.s_max, config.num_anchors, config.map_size,
                                   config.stride_size, )

        self.anchors = tf.constant(self.anchors)

    def handle_im(self, im_input, size):
        im = tf.image.decode_jpeg(im_input, channels=3)

        h = tf.shape(im)[0]
        w = tf.shape(im)[1]
        im = tf.image.resize_images(im, [size, size], method=0)
        im = tf.to_float(im)
        h = tf.to_float(h)
        w = tf.to_float(w)
        sh = h / size
        sw = w / size
        box_scale = tf.concat([[sh], [sw], [sh], [sw]], axis=0)
        im = im[None]

        return im, box_scale
        pass

    def build_net(self, ):
        self.im_input = tf.placeholder(tf.string, name='input')
        im, b_scale, = self.handle_im(self.im_input, size=self.config.img_size)
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
            var_pre = tf.global_variables()
            print(len(var_pre))
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
            for i in (net4, net7, net8, net9, net10, net11):
                print(i)
            with tf.variable_scope('detect'):
                with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
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

        pre_bboxes = loc2bbox(net[..., -4:] * tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32), self.anchors)
        pre = predict(pre_bboxes[0], net[0][..., :-4], size=self.config.img_size,c_thresh=1e-2)
        self.result = tf.concat([pre[..., :4] * b_scale, pre[..., 4:]], axis=-1, name='out')
        print(self.result)

    def test(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
            self.build_net()

        file = self.config.pre_model
        saver = tf.train.Saver()
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, file)
            Res = {}
            i = 0
            m = 100000000
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res = sess.run(self.result, feed_dict={self.im_input: img})
                res = res[:, [1, 0, 3, 2, 4, 5]]
                Res[name] = res
            print(datetime.now() - time_start)
            joblib.dump(Res, 'SSD300_2x.pkl')

        cv_Map.eval(Res, m, True)
        cv_Map.eval(Res, m, False)

    def save_pb(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
            self.build_net()

        file = '/home/zhai/PycharmProjects/Demo35/SSD/train/models/SSD300_2x.ckpt-120000'
        # file='/home/zhai/PycharmProjects/Demo35/myDNN/SSD_tf/train/models/SSD300_4.ckpt-60000'
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, file)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])
            with tf.gfile.FastGFile('model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def test_pb(self):
        pb_file = 'model.pb'
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)
        sess = tf.Session()
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            constant_values = {}
            constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
            for constant_op in constant_ops:
                print(constant_op.name, )
            # 需要有一个初始化的过程
            #     sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('input:0')
            out = sess.graph.get_tensor_by_name('out:0')

            Res = {}
            i = 0
            m = 10000000
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res = sess.run(out, feed_dict={input_x: img})
                res = res[:, [1, 0, 3, 2, 4, 5]]
                Res[name] = res
            print(datetime.now() - time_start)
            joblib.dump(Res, 'SSD300_2x.pkl')

        cv_Map.eval(Res, m, True)
        cv_Map.eval(Res, m, False)


import cv_Map

if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)

    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']

    num_anchors = [4, 6, 6, 6, 4, 4]
    map_size = [38, 19, 10, 5, 3, 1]
    stride_size = [8, 16, 32, 64, 100, 300]
    pre_model = '/home/zhai/PycharmProjects/Demo35/SSD/train/models/SSD300_2x.ckpt-120000'
    config = Config(False, Mean, files, pre_model, s_min=0.2, img_size=300, batch_size_per_GPU=16, gpus=2,
                    weight_decay=0.0005,
                    jitter_ratio=[0.3, 0.5, 0.7], crop_iou=0.45, keep_ratio=0.2, img_scale_size=[212, 150, 106, 75],
                    num_anchors=[4, 6, 6, 6, 4, 4],
                    stride_size=[8, 16, 32, 64, 100, 300], map_size=[38, 19, 10, 5, 3, 1])
    ssd = SSD(config)
    ssd.test()
    # ssd.save_pb()
    # ssd.test_pb()
    pass
