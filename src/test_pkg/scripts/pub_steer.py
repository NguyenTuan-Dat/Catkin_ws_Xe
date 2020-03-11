#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32

msg_steer = Float32()
pub = rospy.Publisher('/set_speed_car_api', Float32, queue_size=10)
rospy.init_node('publish-steer', anonymous=True)


msg_steer.data = 0

while True:
    print(msg_steer.data)
    pub.publish(msg_steer)
    rospy.sleep(2)
    msg_steer.data = 10 - msg_steer.data