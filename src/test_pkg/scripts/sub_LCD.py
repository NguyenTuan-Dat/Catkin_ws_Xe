#!/usr/bin/env python

import rospy
from std_msgs.msg import String


#init ros node
pub = rospy.Publisher('lcd_print', String, queue_size=10)
rospy.init_node('publish-2', anonymous=True)

Xau = "ABC"
while True:
    print(Xau)
    pub.publish(Xau)
    rospy.sleep(2)
   