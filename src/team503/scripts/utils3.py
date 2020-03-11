import cv2
import numpy as np
import time
from SegProcessing import get_road_mask
import seaborn as sns
import matplotlib.pyplot as plt
road_lower = []
road_upper = []

snow_lower = []
snow_upper = []

shadow_lower = []
shadow_upper = []

def get_road(img):
    result = cv2.inRange(img, road_lower, road_upper)
    return result

def get_snow(img):
    result = cv2.inRange(img, snow_lower, snow_upper)
    return result

def get_shadow(img):
    result = cv2.inRange(img, snow_lower, snow_upper)
    return result

def get_road_range(segment_img, img_hsv):
    masked = cv2.bitwise_and(img_hsv, segment_img)
    return  upper, lower

def sum_img(segment_img, road_img):

    return final_road_segment

def get_dynamic_range(img_hsv, pr_mask):

    return lower, upper
import os

dir = "/home/vietphan/Documents/data/IMG/"
img_names = os.listdir(dir)
img_names.sort()
# print(img_names)
start_time = time.time()
for i in range(1200,1300):
  
    img_path = dir + str(i)+".png"
    img_rgb = cv2.imread(img_path)
    img_mask = cv2.imread(img_path.replace("IMG", "Segmented"))
    road_seg = [128, 64, 128]


    # cv2.imshow("rgb", img_rgb)
    cv2.imshow("mask", img_mask)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    start_time = time.time()
    road_mask = get_road_mask(img_mask)
    cv2.imshow("mask", road_mask)

    road_mask = np.stack([road_mask, road_mask, road_mask], axis= 2)
    masked_road = cv2.bitwise_and(img_hsv, road_mask)
    masked_road = cv2.resize(masked_road,(320,320))
    masked_road = masked_road.astype(np.uint8)
    # print(masked_road.shape)

    m = []
    std = []
    for i in range(0, 3):
        arr = masked_road[:, :, i].flatten()
        arr = arr[arr != 0]
        m.append(np.median(arr))
        std.append(np.std(arr))
        # print(m, std)
        # sns.distplot(arr)

    m = np.array(m)
    std = np.array(std)
    road_lower = m - 2 * std
    road_upper = m + 2 * std
    print(road_lower)
    print(road_upper)
    print("")
    reconstruct = get_road(img_hsv)
    print(time.time() - start_time)
    # reconstruct = cv2.morphologyEx(reconstruct, cv2.MORPH_CLOSE, kernel=np.ones((5,5)))
    cv2.imshow("reconstruct", reconstruct)
    cv2.imshow("masked", masked_road)
    cv2.imshow("img_rgb", img_rgb)
    # plt.show()
    cv2.waitKey(0)