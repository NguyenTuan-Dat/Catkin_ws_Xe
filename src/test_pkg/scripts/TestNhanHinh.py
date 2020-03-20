#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
rospy.init_node('control2', anonymous=True)

def drive_callback(rgb_data):
    np_arr = np.fromstring(rgb_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow("img",image_np)
    cv2.waitKey(1)


def listener():
    rospy.Subscriber('/camera/rgb/image_raw/compressed', CompressedImage, drive_callback, buff_size=2 ** 24)
    # rospy.Subscriber('bt1', Bool, start_cb,buff_size=10)
    # rospy.Subscriber('bt2', Bool, stop_cb,buff_size=10)
    rospy.spin()


if __name__ == '__main__':
    listener()
