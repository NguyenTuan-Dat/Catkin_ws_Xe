import tensorflow as tf
import cv2
import numpy as np
import os
import time
from tensorflow.python.compiler.tensorrt import trt_convert as trt

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
PATH = "/home/ubuntu/Downloads/Data_Real/"
LABEL_COLOR = [[0,0,0],[0,0,255], [255,255,255], [255,0,0], [0,255,0]]
def one_hot_tensors_to_color_images(img_np):
    outputs = []
    output = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype='uint8')
    img_np[np.where(img_np == img_np.max(axis=-1, keepdims=1))] = 1
    for i in range(5):
        indices = np.where(img_np[0,:, :, i] == 1)
        output[indices] = LABEL_COLOR[i]
    outputs.append(output)
    return outputs
#ce = Cross Entropy
def loss_ce(label, predict):
    loss = 0
    for batch in range(BATCH_SIZE):
        for channelId in range(5):
            loss = loss - tf.reduce_mean(tf.cast(label[batch,:,:,channelId], dtype= tf.float16)*tf.log(predict[batch,:,:,channelId]))
    return loss/BATCH_SIZE
def l2(label, predict):
    return tf.reduce_mean((tf.cast(label, dtype= tf.float16)-predict)**2)
def one_hot(label_gray):
    label_channel_list = []
    for class_id in range(5):
        equal_map = tf.equal(label_gray, class_id)
        binary_map = tf.to_int32(equal_map)
        label_channel_list.append(binary_map)
    label_xin = tf.stack(label_channel_list, axis=3)
    label_xin = tf.squeeze(label_xin, axis=-1)
    return label_xin

def encoder_block(input_, features, k_size=3, stride=1):
    conv = tf.layers.conv2d(inputs=input_, filters=features, kernel_size=k_size, strides=stride, padding="same", activation="relu")
    # conv = tf.layers.conv2d(inputs=conv, filters=features, kernel_size=k_size, strides=stride, padding="same", activation="relu")
    max_pool = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, padding="same")
    bn = tf.layers.batch_normalization(max_pool)
    return bn

def decoder_block(input_, block_in, features, k_size=3):
    up_conv = tf.layers.conv2d_transpose(inputs=input_, filters=features, kernel_size=2, strides=2, padding="same")
    concat = tf.concat([block_in, up_conv], axis=-1)
    # convUP = tf.layers.conv2d(inputs=concat, filters=features, kernel_size=k_size, strides=1, padding="same", activation="relu")
    # convUP = tf.layers.conv2d(inputs=convUP, filters=features, kernel_size=k_size, strides=1, padding="same", activation="relu")

    return concat


def Unet(input_image):
    block1 = encoder_block(input_image, features=64)
    block2 = encoder_block(block1, features=128)
    block3 = encoder_block(block2, features=256)
    block4 = encoder_block(block3, features=512)
    block5 = encoder_block(block4, features=1024)

    up_block1 = decoder_block(block5, block4, 512)
    up_block2 = decoder_block(up_block1,  block3, 256)
    up_block3 = decoder_block(up_block2, block2, 128)
    up_block4 = decoder_block(up_block3, block1, 64)

    up_conv5 = tf.layers.conv2d_transpose(inputs=up_block4, filters=32, kernel_size=(3, 3), strides=2, padding="same")
    # Shape = (N, 256, 256, 32)
    # up_conv5 = tf.layers.conv2d_transpose(inputs=up_conv5, filters=32, kernel_size=(3, 3), strides=2, padding="same")
    # # Shape = (N, 256, 256, 32)
    # up_conv5 = tf.layers.conv2d(inputs=up_conv5, filters=32, kernel_size=(3, 3), strides=1, padding="same")
    # # Shape = (N, 256, 256, 32)
    output = tf.layers.conv2d(inputs=up_conv5, filters=5, kernel_size=(3, 3), strides=1, padding="same")
    # Shape = (N, 256, 256, 6)

    output = tf.nn.softmax(output)

    return output

def Unet_classify(input_image):
  '''
  :param input_image: Input shape = (N, 480, 480, 1)
  :return:
  '''
  # block1
  conv1_1 = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 480, 480, 64)
  # block2
  maxP2 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2, padding="same")
  conv2_1 = tf.layers.conv2d(inputs=maxP2, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  #Shape = (N, 240, 240, 128)
  # block3
  maxP3 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=(2, 2), strides=2, padding="same")
  conv3_1 = tf.layers.conv2d(inputs=maxP3, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 120, 120, 256)
  # block4
  maxP4 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=(2, 2), strides=2, padding="same")
  conv4_1 = tf.layers.conv2d(inputs=maxP4, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 60, 60, 512)
  # block5
  maxP5 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=(2, 2), strides=2, padding="same")
  conv5_1 = tf.layers.conv2d(inputs=maxP5, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
  conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=1024, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 30, 30, 1024)
  #================#
  #  SEGMENTATION  #
  #================#
  #blockUP1
  up_conv1 = tf.layers.conv2d_transpose(inputs=conv5_2, filters=512, kernel_size= (2, 2), strides=2, padding="same")
  # Shape = (N, 60, 60, 512)
  concat1 = tf.concat([conv4_2, up_conv1], axis=3)
  # Shape = (N, 60, 60, 1024)
  convUP1_1 = tf.layers.conv2d(inputs=concat1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  convUP1_2 = tf.layers.conv2d(inputs=convUP1_1, filters=512, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 60, 60, 512)
  #blockUP2
  up_conv2 = tf.layers.conv2d_transpose(inputs=convUP1_2, filters=256, kernel_size=(2, 2), strides=2, padding="same")
  # Shape = (N, 120, 120, 256)
  concat2 = tf.concat([conv3_2, up_conv2], axis=3)
  # Shape = (N, 120, 120, 512)
  convUP2_1 = tf.layers.conv2d(inputs=concat2, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  convUP2_2 = tf.layers.conv2d(inputs=convUP2_1, filters=256, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 120, 120, 256)
  #blockUP3
  up_conv3 = tf.layers.conv2d_transpose(inputs=convUP2_2, filters=128, kernel_size=(2, 2), strides=2, padding="same")
  # Shape = (N, 240, 240, 128)
  concat3 = tf.concat([conv2_2, up_conv3], axis=3)
  # Shape = (N, 240, 240, 256)
  convUP3_1 = tf.layers.conv2d(inputs=concat3, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  convUP3_2 = tf.layers.conv2d(inputs=convUP3_1, filters=128, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 240, 240, 128)
  #blockUP4
  up_conv4 = tf.layers.conv2d_transpose(inputs=convUP3_2, filters=64, kernel_size=(2, 2), strides=2, padding="same")
  # Shape = (N, 480, 480, 64)
  concat4 = tf.concat([conv1_2, up_conv4], axis=3)
  # Shape = (N, 480, 480, 128)
  convUP4_1 = tf.layers.conv2d(inputs=concat4, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  convUP4_2 = tf.layers.conv2d(inputs=convUP4_1, filters=64, kernel_size=(3, 3), strides=1, padding="same")
  # Shape = (N, 480, 480, 64)
  output_segmentation = tf.layers.conv2d(inputs= convUP4_2, filters=5, kernel_size=(3, 3), strides=1, padding="same")
  # #==============#
  # #    Classify  #
  # #==============#
  # flatten = tf.layers.flatten(inputs=conv5_2)
  # #Shape = (N, 1024*30*30)
  # fully = tf.layers.dense(inputs=flatten, units=1024)
  # fully = tf.layers.dense(inputs=fully, units=512)
  # fully = tf.layers.dense(inputs=fully, units=256)
  # fully = tf.layers.dense(inputs=fully, units=128)
  # fully = tf.layers.dense(inputs=fully, units=3)
  # output_classification = tf.nn.softmax(fully)
  return output_segmentation
      # , output_classification

list_label = os.listdir(PATH + 'labels/')
#
input_tensor = tf.placeholder(dtype=tf.float32, shape=(1, IMG_HEIGHT, IMG_WIDTH, 1))
label_gray_tensor = tf.placeholder(dtype=tf.float32, shape=(1, IMG_HEIGHT, IMG_WIDTH, 1))
label_tensor = one_hot(label_gray_tensor)
predict_tensor = Unet_classify(input_tensor)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.Saver().restore(sess, "/home/ubuntu/Downloads/Model_CDS-20200304T082856Z-001/Model_CDS/model_CDS_Unet.ckpt")
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
label = cv2.imread(PATH + "labels/" + list_label[29])
label = cv2.resize(label, (256,256))
label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
label = np.expand_dims(label, axis=0)
label = np.expand_dims(label, axis=-1)
BGR = cv2.imread(PATH + "RGBs/" + list_label[29])
BGR = cv2.resize(BGR, (256, 256))
image = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=-1)
image = image/255.0
result, label_val = sess.run([predict_tensor, label_tensor], feed_dict={input_tensor: image, label_gray_tensor: label})
result, label_val = sess.run([predict_tensor, label_tensor], feed_dict={input_tensor: image, label_gray_tensor: label})
result, label_val = sess.run([predict_tensor, label_tensor], feed_dict={input_tensor: image, label_gray_tensor: label})
t= time.time()
result, label_val = sess.run([predict_tensor, label_tensor], feed_dict={input_tensor: image, label_gray_tensor: label})
print(time.time()-t)
cv2.imshow("Predict", one_hot_tensors_to_color_images(result)[0])
cv2.imshow("Label", one_hot_tensors_to_color_images(label_val)[0])
cv2.imshow("RGB", BGR)
cv2.waitKey()