#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, String, Bool
from sensor_msgs.msg import CompressedImage
import os
from pathlib import Path
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


pub_img = rospy.Publisher('/camera/rgb/image_raw/compressed', CompressedImage)
rospy.init_node('control', anonymous=True) 


data_RGB = "/home/ubuntu/data/RGB"

imgs = sorted(Path(data_RGB).iterdir(), key=os.path.getmtime)
for link in imgs:
    img = cv2.imread(data_RGB+"/"+link.name)
    print(link)
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
    pub_img.publish(msg)
    rospy.sleep(0.02)