#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @ 2019 Liming Liu     HuNan University
#


"""predict the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import cv2
import numpy as np
import time
import rospkg
rospack = rospkg.RosPack()
cur_dir = rospack.get_path('team503')

colors = np.array([[0,0,0],
[220,220,0],
[128, 64, 128],
[0, 0, 142],
[70,130,180]], dtype=np.float32)


class BiseNet_Loader(object):
    def __init__(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=sess_config)
        
        with tf.gfile.GFile("/home/ubuntu/cds-segmentation/frozen_bisenet/frozen_model.pb", 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,
            nodes_blacklist=['combine_path/Softmax:0']) #output nodes
        trt_graph = converter.convert()
        # Import the TensorRT graph into a new graph and run:
        self.response = tf.import_graph_def(
            trt_graph,
            return_elements=['combine_path/Softmax:0'])

        # sess.run(local_variables_init_op)
        img = np.ones((1, 256, 256, 3))
        self.transform = tf.reshape(tf.matmul(tf.reshape(tf.one_hot(tf.argmax(self.response, -1), 5), [-1, 5]), colors),
                            [-1, 256, 256, 3])
        _ = self.sess.run(self.transform, feed_dict={"import/image_input:0": img})

        print("Model loaded!")

    def predict(self, img):
        img = cv2.resize(img, (256, 256))
        img = img/255.
        img = np.expand_dims(img, axis=0)
        predict = self.sess.run(self.transform, feed_dict={"import/image_input:0": img})
        predict = cv2.cvtColor(predict[0], cv2.COLOR_RGB2BGR)
        predict = cv2.resize(predict, (480, 640))
        return predict

