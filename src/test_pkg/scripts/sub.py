#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool


#init ros node
pub = rospy.Publisher('led_status', Bool, queue_size=10)
rospy.init_node('publish', anonymous=True)


Xau =  Bool()
Xau.data = True
while True:
    print(Xau.data)
    pub.publish(Xau)
    rospy.sleep(2)
    Xau.data = not Xau.data