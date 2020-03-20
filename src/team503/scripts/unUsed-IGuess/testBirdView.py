import numpy as np
import cv2
img = cv2.imread("/home/denvi/Downloads/data_Viet/data_Viet/1152_color_mask.png")
import time
from LineIterator import get_line_cor

left_cors = get_line_cor((619, 159), (0, 120)).T
right_cors = get_line_cor((619, 161), (0, 200)).T
sub_left_cors = get_line_cor((590, 159), (585, 0)).T
sub_right_cors = get_line_cor((590, 161), (585, 319)).T
turn_thresh = 0
angle_bias = 0.5

def get_distance_to_obstacles(vector, line):
    if not len(vector[0]) == 0:
        return vector[0][0]
    else:
        return line.shape[1]
def get_confident_vectors(bird_view_img):
    angle = 15
    left =  np.where(bird_view_img[left_cors[0], left_cors[1]] == 0)
    right =  np.where(bird_view_img[right_cors[0], right_cors[1]] == 0)
    sub_left =  np.where(bird_view_img[sub_left_cors[0], sub_left_cors[1]] == 0)
    sub_right =  np.where(bird_view_img[sub_right_cors[0], sub_right_cors[1]] == 0)


    left = get_distance_to_obstacles(left, left_cors)
    right = get_distance_to_obstacles(right, right_cors)
    sub_left = get_distance_to_obstacles(sub_left, sub_left_cors)
    sub_right = get_distance_to_obstacles(sub_right, sub_right_cors)


    left += 5 * sub_left
    right += 5 * sub_right
    turn = ((left / (left + right)) - angle_bias) * 2
    return - turn
def get_bird_view(img):
    IMAGE_H = 320
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[140, IMAGE_W], [180, IMAGE_W], [0, 0], [IMAGE_H , 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    img2 = img[140:(120+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img2, M, (IMAGE_H, IMAGE_W-20)) # Image warping
    # cv2.imshow("img", img)
    # cv2.imshow("img2", img2)
    # cv2.imshow("warped_img", warped_img)
    # cv2.waitKey(0)
    return warped_img

line_color_upper = (180,130,70)
line_color_lower = (51,37,26)
 

def get_road_mask(img):
    road_color = (128, 64, 128)
    kernel = np.ones((5,5), np.uint8)
    road = cv2.inRange(img, road_color, road_color)
    road = cv2.dilate(road, kernel, iterations=1) 
    return road
start = time.time()
img_bird = get_bird_view(img)
road_mask = get_road_mask(img_bird) 

print(get_confident_vectors(road_mask)*100)
# print(time.time()-start)
cv2.imshow("road_mask", road_mask)
cv2.waitKey(0)