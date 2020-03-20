#!/usr/bin/env python3
import cv2 
import numpy as np 
from keras.models import model_from_json
import tensorflow as tf 

# from scipy import signal
from LineIterator import get_line_cor
import rospkg
rospack = rospkg.RosPack()
cur_dir = rospack.get_path('team503')

# graph = tf.get_default_graph()
# json_file = open(cur_dir + '/scripts/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(cur_dir + "/scripts/model.h5")
#
# def init_classifier():
#     with graph.as_default():
#         model.predict(np.ones((1,32,32,3)))
#     print("Loaded model from disk")



def dynamic_speed(angle):

    return 30 - (abs(angle) * 0.1)

def get_bird_view(img):

    IMAGE_H = 320
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[140, IMAGE_W], [180, IMAGE_W], [0, 0], [IMAGE_H , 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    img2 = img[140:(120+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img2, M, (IMAGE_H, IMAGE_W-20)) # Image warping
    # cv2.imshow("img", img)
    # cv2.imshow("img2", img2)
    # cv2.imshow("warped_img", warped_img)
    # cv2.waitKey(0)
    return warped_img

car_color = (143, 0, 0)
truck_color = (69, 0, 0)
def get_car_mask(img):

    result = cv2.inRange(img, truck_color, car_color)
    
    # kernel = np.ones((21, 21))
    # result = cv2.dilate(result, kernel)
    # kernel = np.zeros((21, 21))
    # kernel[10, 10] =  1
    # kernel[5, 20] = 1
    # kernel[15, 20] = 1
    # kernel = kernel.astype(np.uint8)
    # result = cv2.dilate(result, kernel)
    # cv2.imshow("sign", result)
    return result

def get_road_mask(img):
    road_color = (128, 64, 128)
    kernel = np.ones((5,5), np.uint8)
    road = cv2.inRange(img, road_color, road_color)
    road = cv2.dilate(road, kernel, iterations=1) 
    return road

def get_roi():
    black = np.zeros([240, 320],dtype=np.uint8)
    pts = np.array([[0,40],[100,40],[120,100],[0, 150]], np.int32)
    cv2.fillPoly(black, [pts], 255)
    pts = np.array([[320,40],[220,40],[200,100],[320, 150]], np.int32)
    cv2.fillPoly(black, [pts], 255)
    return black

roi = get_roi()

sign_color = (0, 220, 220)
def get_sign_mask(img):

    thresh = cv2.inRange(img, sign_color, sign_color)

    
    thresh = cv2.bitwise_and(thresh, roi)

    sign = -1
    # cv2.imshow("sign", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        # print(cv2.contourArea(contours[0]))
        if cv2.contourArea(contours[0]) > 70:
            sign = cv2.boundingRect(contours[0])
    return sign
i = 0
def binary_sign(gray_sign):
    global graph
    global i
    with graph.as_default():        
        # sign = cv2.cvtColor(gray_sign, cv2.COLOR_RGB2GRAY)
        sign = np.array([np.reshape(gray_sign, (32,32,3))])
        result = model.predict(sign, batch_size = 1) + 1
    cv2.imwrite("/home/sonduong/catkin_ws/src/beginner_tutorials/scripts/sign_data_temp/" + str(i) + str(result) + ".png", gray_sign)
    i += 1
    return result

def sign_classify(img_rgb, mask):
    # global graph
    sign_rect = get_sign_mask(mask)
    result = -1
    
    if(sign_rect != -1):
        # with graph.as_default():
        sign = img_rgb[sign_rect[1]: sign_rect[1] + sign_rect[3], sign_rect[0]: sign_rect[0] + sign_rect[2]]
        sign = cv2.resize(sign, (32, 32))
        # sign = cv2.cvtColor(sign, cv2.COLOR_RGB2BGR)
        # sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("sign", sign)
        sign = np.array([sign])
        result = model.predict(sign, batch_size = 1) + 1
    return result

#################### INIT CONFIDENCE VECTOR #########################
left_cors = get_line_cor((619, 159), (0, 120)).T
right_cors = get_line_cor((619, 161), (0, 200)).T
sub_left_cors = get_line_cor((590, 159), (585, 0)).T
sub_right_cors = get_line_cor((590, 161), (585, 319)).T
turn_thresh = 0
angle_bias = 0.5
#################### INIT CONFIDENCE VECTOR #########################

turn_dis_graph = []
turn_dis_cur = 0
def turn_status():
    global turn_dis_graph
    global turn_dis_cur
    mean = np.array(turn_dis_graph).mean()
    if (mean - 10) < turn_dis_cur < (mean + 10):
        return 0
    elif turn_dis_cur >= mean + 10:
        return 1
    elif turn_dis_cur <= mean -10:
        return 2
def get_distance_to_obstacles(vector, line):
    if not len(vector[0]) == 0:
        return vector[0][0]
    else:
        return line.shape[1]


def start_turning(bird_view_img, direction):
    # global angle_bias
    # global left_cors
    # global right_cors
    dir = (direction + 1) * 159
    turn_left_cors = get_line_cor((139, 159), (90, dir)).T
    turn_right_cors = get_line_cor((159, 160), (120, dir)).T    
    # turn_cors = get_line_cor((90, 159), (70, 0)).T
    # angle_bias = 
    left =  np.where(bird_view_img[turn_left_cors[0], turn_left_cors[1]] == 0)
    right =  np.where(bird_view_img[turn_right_cors[0], turn_right_cors[1]] == 0)
    # turn = np.where(bird_view_img[turn_cors[0], turn_cors[1]] == 0)


    left = get_distance_to_obstacles(left, turn_left_cors)
    right = get_distance_to_obstacles(right, turn_right_cors)
    if left == 0:
        left = left_cors.shape[1]
    turn = ((left / (left + right)) - angle_bias) * 2
    # print(left, right)
    return turn * 40

def get_confident_vectors(bird_view_img):
    angle = 15
    # discrete_line = get_line_cor((159, 159), (0, 140), bird_view_img)
    # print(discrete_line.T)
    left =  np.where(bird_view_img[left_cors[0], left_cors[1]] == 0)
    right =  np.where(bird_view_img[right_cors[0], right_cors[1]] == 0)
    sub_left =  np.where(bird_view_img[sub_left_cors[0], sub_left_cors[1]] == 0)
    sub_right =  np.where(bird_view_img[sub_right_cors[0], sub_right_cors[1]] == 0)


    left = get_distance_to_obstacles(left, left_cors)
    right = get_distance_to_obstacles(right, right_cors)
    sub_left = get_distance_to_obstacles(sub_left, sub_left_cors)
    sub_right = get_distance_to_obstacles(sub_right, sub_right_cors)


    # l, r, t = start_turning(bird_view_img)
    left += 5 * sub_left
    right += 5 * sub_right
    turn = ((left / (left + right)) - angle_bias) * 2


    return - turn

turning_frame = 0
turn_dir = 0
list_signs = []
import time
kernel = np.zeros((35, 5))
kernel = np.concatenate((kernel, np.ones((35, 25))), axis= 1)
kernel = np.concatenate((kernel, np.zeros((35, 5))), axis= 1)
kernel = kernel.astype(np.uint8)
def get_sign(img):
    return -1


def get_steer(sign, mask):
    # global turn_thresh
    global turning_frame
    global turn_dir
    global list_signs
    # start_time = time.time()

    frame = cv2.resize(mask, (640, 480)).astype(np.uint8)

    # print(start_time - time.time())
    # start_time = time.time()

    road_mask = get_road_mask(frame)

    # print(time.time() - start_time)
    # start_time = time.time()
    bird_view = get_bird_view(road_mask)
    # print(time.time() - start_time)
    # start_time = time.time()
    # car_mask = get_car_mask(frame)
    # print(time.time() - start_time)
    # start_time = time.time()

    # car_mask = get_bird_view(car_mask)
    # print(time.time() - start_time)
    # start_time = time.time()
    # car_mask = cv2.dilate(car_mask, kernel)
    # temp = cv2.bitwise_and(car_mask, bird_view)
    # print(time.time() - start_time)
    # start_time = time.time()
    # bird_view = bird_view - temp
    bird_view = bird_view.astype(np.uint8)

    # cv2.imshow("mask", frame)
    # cv2.imshow("bird_view", bird_view)
    angle = (get_confident_vectors(bird_view)) * 100
    speed = dynamic_speed(angle)

    if sign != -1:
        print("Co bien bao! " + str(sign))
        # sign = sign[0][0]
        list_signs.append(sign)
        sign = np.array(list_signs).mean()
        sign = int(round(sign))
        turning_frame = 55
        speed = 40
        turn_dir = sign - 1
        if turn_dir == 0:
            turn_dir = 1
            # elif turn_dir == 1:
        #     turn_dir = -1

    if turning_frame > 0:
        # turning ...
        turning_frame -= 1
        angle_bias = (start_turning(bird_view, turn_dir))
        print(angle_bias)
        if angle_bias < 0:
            angle_bias = 0
        angle = (angle_bias * turn_dir) + angle
        speed = 40
    else:
        turn_dir = 0
        list_signs = []
    # print(start_time - time.time())
    # start_time = time.time()
    # print("")

    return speed, angle


def get_steer2(sign, mask):
    # global turn_thresh
    global turning_frame
    global turn_dir
    global list_signs
    # start_time = time.time()

    frame = cv2.resize(mask, ( 320, 240)).astype(np.uint8)
    # print(start_time - time.time())
    # start_time = time.time()

    road_mask = get_road_mask(frame)

    # print(time.time() - start_time)
    # start_time = time.time()
    bird_view = get_bird_view(road_mask)
    # print(time.time() - start_time)
    # start_time = time.time()
    car_mask = get_car_mask(frame)
    # print(time.time() - start_time)
    # start_time = time.time()

    car_mask = get_bird_view(car_mask)
    # print(time.time() - start_time)
    # start_time = time.time()
    car_mask = cv2.dilate(car_mask, kernel)
    temp = cv2.bitwise_and(car_mask, bird_view)
    # print(time.time() - start_time)
    # start_time = time.time()
    bird_view = bird_view - temp
    bird_view = bird_view.astype(np.uint8)

    # cv2.imshow("mask", frame)
    # cv2.imshow("bird_view", bird_view)
    angle = (get_confident_vectors(bird_view)) * 25
    speed = dynamic_speed(angle)

    if sign != -1:
        print("Co bien bao! " + str(sign))
        sign = sign[0][0]
        list_signs.append(sign)
        sign = np.array(list_signs).mean()
        sign = int(round(sign))
        turning_frame = 55
        speed = 40
        turn_dir = sign - 1
        if turn_dir == 0:
            turn_dir = -1 
        # elif turn_dir == 1:
        #     turn_dir = -1

    if turning_frame > 0:
        #turning ...
        turning_frame -= 1
        angle_bias = (start_turning(bird_view , turn_dir))
        print(angle_bias)
        if angle_bias < 0:
            angle_bias = 0
        angle = (angle_bias * turn_dir) + angle
        speed = 40
    else:
        turn_dir = 0
        list_signs = []
    # print(start_time - time.time())
    # start_time = time.time()
    # print("")
    
    return  speed, angle

