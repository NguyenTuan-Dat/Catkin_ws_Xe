#!/usr/bin/env python3
import roslib
roslib.load_manifest('beginner_tutorials')
import sys
# import tensorflow
import rospy
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
rospy.init_node('control2', anonymous=True)
import time
import base64
import numpy as np
import cv2

def callback(ros_data):
	status = ros_data.data 

	jpg_original = base64.b64decode(status)
	# print(status)
	jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
	img = cv2.imdecode(jpg_as_np, flags=1)
	cv2.imshow("bird_view", img)
	cv2.waitKey(1)

def drive_callback(ros_data):

	if(time.time()-start_time > 0.01):
		np_arr = np.fromstring(ros_data.data, np.uint8)
		image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAY)
		# image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		cv2.imshow('TESTING', image_np)
		cv2.waitKey(1)

def listener():
	rospy.Subscriber('team1/birdView', String, callback)
	rospy.spin()

if __name__ == '__main__':
    listener()
