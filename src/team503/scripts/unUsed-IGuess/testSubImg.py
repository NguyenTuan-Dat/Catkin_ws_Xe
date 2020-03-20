#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
import numpy as np
import cv2
from utils import preprocess
# from tensorflow.keras import load_model
# from model import build_model, get_seg
import message_filters
import math
# from SegProcessing import get_steer, binary_sign, init_classifier
from SegPro_2 import get_steer2, get_bird_view, get_road_mask
v0 = 50
a0 = 0
status = 0
time_interval = 0.1
num_pic = 3

check = 0
previous_states = []


pub_steer = rospy.Publisher('team1/set_angle', Float32, queue_size=10)
pub_speed = rospy.Publisher('team1/set_speed', Float32, queue_size=10)

msg_speed = Float32()
msg_speed.data = 50
msg_steer = Float32()
msg_steer.data = 0

def move_img(img,state):
    global time_interval
    global num_pic
    height, width = img.shape[:2] 
    e1 = -7.12
    e2 = 1.6
    v = state[0]
    a = state[1]
    time_between = time_interval/num_pic*0.95
    x_change = v*e2*time_between*math.cos((180-a*e1)*3.14/180)
    y_change = v*e2*time_between*math.sin((180-a*e1)*3.14/180)
    print("Dich chuyen")
    print(x_change,y_change)
    T = np.float32([[1, 0, x_change], [0, 1, y_change]]) 
    img_translation = cv2.warpAffine(img, T, (width, height)) 
    return img_translation
def statusCallback(ros_data):
    global status
    status = ros_data.data
    print(status)
road_color = (128, 64, 128)
def callback(data):
    global previous_states
    global status
    global check
    br = CvBridge()
    print("Co hinh ne")
    flag = status
    v_temp = 50
    a_temp = 0
    rospy.loginfo('receiving image')
    img = br.imgmsg_to_cv2(data)
    cv2.imshow("camera",img)
    # Move the img 
    # if(img.all()==0):
    #     check = 0
    img = cv2.resize(img, ( 320, 240)).astype(np.uint8)
    # img = get_road_mask(img)
    img = get_bird_view(img)



    
    time_between = time_interval/num_pic*0.95

    # Generate predictions:
    if(check!=0):
        print(previous_states)
        for i in range(len(previous_states)):
            if(not math.isnan(previous_states[i][0])):
                img = move_img(img,previous_states[i])
        cv2.imshow("Shifted",img)
        previous_states = []
        for i in range(0,num_pic):
            v_temp,a_temp = get_steer2(-1,img)
            img = move_img(img,(v_temp,a_temp)) 
            cv2.imshow("Processing",img)
            msg_steer.data = float(a_temp)
            msg_speed.data = float(v_temp)
            pub_steer.publish(msg_steer)
            pub_speed.publish(msg_speed)
            print("Thong so: ")
            print(a_temp,v_temp)
            #Publish a_temp,v_temp
            previous_states.append((v_temp,a_temp))
            rospy.sleep(time_between)
        
    else:
        for i in range(0,num_pic):
            msg_steer.data = float(a0)
            msg_speed.data = float(v0)
            pub_steer.publish(msg_steer)
            pub_speed.publish(msg_speed)
            previous_states.append((v0,a0))
            rospy.sleep(time_between)
        check+=1

    cv2.waitKey(1)
def listener():
    rospy.init_node('imager',anonymous=True)
    # rospy.Subscriber('team1/birdViewImg',Image, callback)
    rospy.Subscriber('team1/birdViewImg', Image,callback)
    # rospy.Subscriber('team1/status', Float32,statusCallback)  
    rospy.spin()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    listener()