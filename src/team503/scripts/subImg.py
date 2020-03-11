#!/usr/bin/env python3
from __future__ import print_function

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import roslib
roslib.load_manifest('beginner_tutorials')
import sys
# import tensorflow
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import Float32, String
from sensor_msgs.msg import CompressedImage, Image
import csv
import time
from cv_bridge import CvBridge, CvBridgeError
import base64


# from utils import preprocess
# # from tensorflow.keras import load_model
# from model import build_model, get_seg
num = 0
num2 = 0
check = True
check2 = True  
PATH = '/home/vietphan/Documents/data/IMG'
PATH2 = '/home/vietphan/Documents/data'


pub_steer = rospy.Publisher('team1/set_angle', Float32, queue_size=10)
pub_speed = rospy.Publisher('team1/set_speed', Float32, queue_size=10)
pub_status = rospy.Publisher('team1/status', Float32, queue_size=10)

pub_birdview = rospy.Publisher('team1/birdView', String, queue_size=1)

rospy.init_node('control', anonymous=True)
msg_speed = Float32()
msg_speed.data = 50
msg_steer = Float32()
msg_steer.data = 0
msg_status = Float32()
msg_status.data = 0
angle = 0
msg = String()
msg.data = ""

# model = build_model()
# # model = get_seg()

# model.load_weights("/home/sonduong/catkin_ws/src/beginner_tutorials/scripts/model.h5")

# model.predict(np.ones((1, 66, 200, 3)), batch_size= 1)
# print("Model loaded !")


# image_np = cv2.imread("/home/sonduong/Documents/data/IMG/0.png")
# # image_np = cv2.resize(image_np, (320, 320))
# image = np.asarray(image_np)       # from PIL image to numpy array
# # image = preprocess(image) # apply the preprocessing
# image = np.array([image])       # the model expects 4D array
# # print(float(model.predict(image, batch_size=1)))
# pr_mask = model.predict(image/255)
# print(pr_mask.shape)
num = 0
num2 = 0
check = True
check2 = True  



start_time = time.time()

def callback(speed,steer):
	global num
	global check
	global check2
	if check2:
		rospy.loginfo('SteerAngle '+str(num))
		#rospy.loginfo(steer)
		rospy.loginfo('Speed '+str(num))
		#rospy.loginfo(speed)
		num=num+1
		check = True
		check2 = False
		with open(PATH2+'data.csv', 'ab') as csvfile:
			dataWriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			dataWriter.writerow([str(num),str(speed.data),str(steer.data),str(time.time())])

def callback2(ros_data):
	global num2
	global check
	global check2
	np_arr = np.fromstring(ros_data.data, np.uint8)
	image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
	cv2.imshow('cv_img', image_np)
	cv2.waitKey(2)
	if check:
		rospy.loginfo('IMG '+str(num2))
		cv2.imwrite(PATH+'/'+str(num2)+'.png', image_np)
		num2 = num2+1
		check = False
		check2 = True
	
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
   

	# rate = rospy.Rate(1)
	# while not rospy.is_shutdown():
	# 	rate.sleep()
	# 	# rospy.Subscriber('team1/set_speed', Float32, callback = print_speed)
	# 	# rospy.Subscriber('team1/set_angle', Float32, callback = print_angle)
	# 	rospy.Subscriber('team1/camera/rgb/compressed', CompressedImage, drive_callback)
		


	# rospy.init_node('listener3', anonymous=True)
    #img_sub = message_filters.Subscriber('Team1_image/compressed', CompressedImage)
    
    #rospy.Subscriber('Team1_speed', Float32, callback2)
    #rospy.Subscriber('Team1_steerAngle', Float32, callback3)  

	speed_sub = message_filters.Subscriber('team1/set_speed', Float32)
	steer_sub = message_filters.Subscriber('team1/set_angle', Float32)
	ts = message_filters.ApproximateTimeSynchronizer([speed_sub, steer_sub], 10, 0.1, allow_headerless=True)
	ts.registerCallback(callback)
	rospy.Subscriber('team1/camera/rgb/compressed', CompressedImage, callback2)
	# rospy.Subscriber('team1/camera/depth/compressed', CompressedImage, depth_callback)
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
    listener()
