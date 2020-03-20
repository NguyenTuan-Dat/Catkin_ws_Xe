#!/usr/bin/env python3
from __future__ import print_function

import roslib

roslib.load_manifest('team503')
import sys
# import tensorflow
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
import csv
import time
import threading
from utils import preprocess
import tensorflow as tf
import math
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session
import rospkg
from sign_classifier import SignClassifier
rospack = rospkg.RosPack()
cur_dir = rospack.get_path('team503')


gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=sess_config)
graph = tf.get_default_graph()

set_session(sess)

import rospkg

json_file = open(cur_dir+"/scripts/model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()

model2 = model_from_json(loaded_model_json)
model2.load_weights(cur_dir+"/scripts/model.h5")

sign_cascade = cv2.CascadeClassifier(cur_dir+"/scripts/cascade.xml")

rospack = rospkg.RosPack()
cur_dir = rospack.get_path('team503')


pub_steer = rospy.Publisher('team503/set_angle', Float32, queue_size=10)
pub_speed = rospy.Publisher('team503/set_speed', Float32, queue_size=10)

rospy.init_node('control', anonymous=True)
msg_speed = Float32()
msg_speed.data = 50
msg_steer = Float32()
msg_steer.data = 0

depth_np = 0
depth_hist = 0

from tf_bisenet.BiSeNet_Loader import BiseNet_Loader
from SegPro_3 import get_steer, sign_classify, get_steer2, get_road_mask
from SegPro_3 import get_bird_view, get_confident_vectors, dynamic_speed, get_car_mask, get_steer_no_mask

print("Den day roi ne")
model = BiseNet_Loader()
print("Den day roi ne 2")

msg_speed = Float32()
msg_speed.data = 20
msg_steer = Float32()
msg_steer.data = 0

start_time = time.time()
lock_key = 0
pr_mask = np.zeros((320, 320, 3))
road_mask = np.zeros((320, 320))
temp_img = np.zeros((240, 320, 3))
bird_view_segmented = np.zeros((240, 320, 3))
obstacles = np.zeros((320, 240))
shift_rate = 8


def extract_mask(image):
    global model
    global lock_key
    global pr_mask
    global temp_img
    global bird_view_segmented
    global check
    global upper
    global lower
    global obstacles
    global road_mask
    lock_key = 1
    check = 1
    pr_mask = model.predict(image)
    img_hsv = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2HSV), (320, 320))
    road_mask = get_road_mask(pr_mask)
    road_mask_ = np.stack([road_mask, road_mask, road_mask], axis=2)
    masked_road = cv2.bitwise_and(img_hsv, road_mask_)
    masked_road = masked_road.astype(np.uint8)
    m = []
    std = []
    for i in range(0, 3):
        arr = masked_road[:, :, i].flatten()
        arr = arr[arr != 0]
        m.append(np.median(arr))
        std.append(np.std(arr))
    m = np.array(m)
    std = np.array(std)

    lower = m - 3 * std
    upper = m + 3 * std

    obstacles_ = get_car_mask(pr_mask)
    kernel = np.ones((15, 15), np.uint8)
    obstacles = cv2.resize(cv2.dilate(obstacles_, kernel, iterations=1), (320, 240))
    lock_key = 0



IMAGE_H = 240
IMAGE_W = 320
src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32(
    [[0 - shift_rate, IMAGE_H + shift_rate], [IMAGE_W + shift_rate, IMAGE_H + shift_rate], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
speed = 10
rows, cols = 320, 320
sign = -1
frame_count = 0
check = 0
lower = np.array([-13.56911506, 0.39137337,
                  41.27434073])  # [74.94252403 -1.1260862  44.66248593] [121.05747597  35.1260862  111.33751407]
upper = np.array([121.05747597, 35.1260862, 111.33751407])
previous_state = (50, 0)

e2 = 20
e1 = 1.6
t = 0.1


def tranform(img, v, a):
    height, width = img.shape[:2]
    sign = 1
    if (a > 0):
        sign = -1
    delta_y = v * e1 * math.cos(abs(a) * e2 * 3.14 / 180) * t
    delta_x = v * e1 * math.sin(abs(a) * e2 * 3.14 / 180) * t
    print(delta_x, delta_y)
    T = np.float32([[1, 0, -sign * delta_x], [0, 1, -delta_y]])
    img_translation = cv2.warpAffine(img, T, (width, height))
    return img_translation

SC = SignClassifier()
SC.init_classifier()
model2 = SC.model
def drive_callback(rgb_data):
    global start_time
    global lock_key
    global pr_mask
    global M
    global shift_rate
    global sign
    global dst
    global speed
    global frame_count
    global check
    global bird_view_segmented
    global previous_state
    global temp_img
    global lower
    global upper
    global obstacles
    global road_mask
    # global graph
    road_seg = [128, 64, 128]
    if (time.time() - start_time > 0.001):

        np_arr = np.fromstring(rgb_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_RGB = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        sign = -1
        image_gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)
        sign_rect = sign_cascade.detectMultiScale(image_gray, 1.13, 1)
        for (x_sign, y_sign, w_sign, h_sign) in sign_rect:
            sign_crop = image_RGB[y_sign:y_sign + h_sign, x_sign:x_sign + w_sign]
            sign_crop = cv2.resize(sign_crop, (16, 16))
            sign_predicted = model2.predict(np.array([sign_crop]))
            print(sign_predicted)
            sign_id = np.where(sign_predicted[0] == np.amax(sign_predicted[0]))
            print(sign_id)
            if sign_id[0] == 1: 
                print("RE PHAIIIIIIIII")
                sign = 1
            elif sign_id[0] == 0: 
                print("RE TRAIIIIIIIII")
                sign = 0
        img_hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        img_ranged = cv2.inRange(img_hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        if (check != 0):
            img_ranged = cv2.subtract(img_ranged, obstacles)

        img_ranged = cv2.dilate(img_ranged, kernel, iterations=1)

        if lock_key == 0:
            t1 = threading.Thread(target=extract_mask, args=(image_RGB,))
            t1.start()
        if (check != 0):
            speed, angle = get_steer_no_mask(sign, img_ranged)
            msg_steer.data = float(angle)
            msg_speed.data = float(speed)
            pub_steer.publish(msg_steer)
            pub_speed.publish(msg_speed)
        cv2.waitKey(1)




def listener():
    rospy.Subscriber('team503/camera/rgb/compressed', CompressedImage, drive_callback, buff_size=2 ** 24)
    rospy.spin()


if __name__ == '__main__':
    listener()
