#!/usr/bin/env python3
from pynput import keyboard
import time
import rospy
from std_msgs.msg import Float32, String, Bool

class MyListener(keyboard.Listener):
    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.go = False
        self.right = False
        self.left = False
        self.back = False
    def on_press(self, key):
        if key == keyboard.Key.up:
            self.go = True
        if key == keyboard.Key.right:
            self.right = True
        if key == keyboard.Key.left:
            self.left = True
        if key == keyboard.Key.down:
            self.back = True
    def on_release(self, key):
        if key == keyboard.Key.up:
            self.go = False
        if key == keyboard.Key.right:
            self.right = False
        if key == keyboard.Key.left:
            self.left = False
        if key == keyboard.Key.down:
            self.back = False

msg_speed = Float32()
msg_speed.data = 0
msg_steer = Float32()
msg_lcd = String()
pub_steer = rospy.Publisher('set_steer_car_api', Float32, queue_size=10)
pub_speed = rospy.Publisher('set_speed_car_api', Float32, queue_size=10)
pub_lcd = rospy.Publisher('lcd_print', String, queue_size=1)
pub_led = rospy.Publisher('led_status', Bool, queue_size=10)
rospy.init_node('control', anonymous=True) 
right_accelerate = 0
left_accelerate = 0
amount = 5
rate = 0.2

listener = MyListener()
listener.start()

started = False

while True:
    time.sleep(rate)
    if listener.go == True:
        msg_speed.data = 60
        pub_speed.publish(msg_speed)
        msg_lcd.data = "ROAD TO HN ..."
        pub_lcd.publish(msg_lcd)
        pub_led.publish(True)
    elif listener.back == True:
        msg_speed.data = -10
        pub_speed.publish(msg_speed)
        msg_lcd.data = "ROAD TO HN ..."
        pub_lcd.publish(msg_lcd)
        pub_led.publish(True)
    elif listener.go == False and listener.back == False:
        msg_speed.data = 0
        pub_speed.publish(msg_speed)
        msg_lcd.data = "stop"
        pub_lcd.publish(msg_lcd)
        pub_led.publish(False)
    if listener.right == True:
        left_accelerate = 0
        msg_steer.data = 50
        right_accelerate = right_accelerate + rate * 13
        pub_steer.publish(msg_steer)
    elif listener.left == True:
        right_accelerate = 0
        msg_steer.data = -50
        left_accelerate = left_accelerate + rate * 13
        pub_steer.publish(msg_steer)
    if listener.right == False and listener.left == False:
        msg_steer.data = 0
        pub_steer.publish(msg_steer)
        right_accelerate = 0
        left_accelerate = 0
	
	
        
