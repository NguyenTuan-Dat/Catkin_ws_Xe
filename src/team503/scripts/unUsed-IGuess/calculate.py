import cv2
import math
import numpy as np
img = cv2.imread("/home/vietphan/Downloads/done_segment_viet/GT/303_color_mask.png",1)
img2 = cv2.imread("/home/vietphan/Downloads/done_segment_viet/GT/273_color_mask.png",1)
cv2.imshow('image', img) 

e2 = 20
e1 = 1.6
t = 0.15
def tranform(img,v,a):
    height, width = img.shape[:2] 
    sign = 1
    if(a>0):
        sign = -1
    delta_y = v*e1*math.cos(abs(a)*e2*3.14/180)*t
    delta_x = v*e1*math.sin(abs(a)*e2*3.14/180)*t
    print(delta_x,delta_y)
    T = np.float32([[1, 0, -sign*delta_x], [0, 1, -delta_y]]) 
    img_translation = cv2.warpAffine(img, T, (width, height)) 
    return img_translation
img_translation = tranform(img,50,-12.8)
cv2.imshow('img_translation', img_translation)
cv2.imshow('GroundTruth', img2)

cv2.waitKey(0)
