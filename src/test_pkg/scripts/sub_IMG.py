#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
import cv2
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
cv_bridge = CvBridge()
rospy.init_node('control', anonymous=True)
def depth_callback(data):
    img = cv_bridge.imgmsg_to_cv2(data, "passthrough")
    cv2.imshow("image_depth",img)
    cv2.waitKey(1)
def drive_callback(rgb_data):
    np_arr = np.fromstring(rgb_data.data, np.uint8)
    
    image_np = cv2.imdecode(np_arr, 1)
    # print(type(image_np))
    # image_RGB = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("image_RGB",image_np)
    # print("HU?")
    cv2.waitKey(1)

def listener():

    # rospy.Subscriber('team1/camera/depth/compressed', CompressedImage, bias_callback)
    rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, drive_callback,  buff_size=2**24)
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback,  buff_size=2**24)

    # rospy.Subscriber('Team1_image/compressed', CompressedImage, drive_callback,  buff_size=2**24)
    # ls = sync_listener()
    # depth_sub = message_filters.Subscriber('team1/camera/depth/compressed', CompressedImage)
    # image_sub = message_filters.Subscriber('team1/camera/rgb/compressed', CompressedImage)

    # ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
    # ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
 
    rospy.spin()

if __name__ == '__main__':
    listener()