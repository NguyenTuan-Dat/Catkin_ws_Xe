#!/usr/bin/env python3
import cv2
import numpy as np

from SegProcessing import get_bird_view
from tf_bisenet.BiSeNet_Loader import BiseNet_Loader
from SegProcessing import get_steer, init_classifier, sign_classify
import os

model = BiseNet_Loader()

dir = "/home/vietphan/Documents/data/IMG/"
dir2 = "/home/vietphan/Documents/data/Segmented/"
img_names = os.listdir(dir)
i = 0
for name in img_names:
    path = dir+name
    img = cv2.imread(path)
    # print(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_predited = model.predict(img)
    img_predited = cv2.resize(img_predited,(320,240))
    cv2.imshow("img_predited",img_predited)
    cv2.waitKey(1)

    # masked_road = cv2.resize(masked_road,(320,240))
    
    cv2.imwrite(dir2+name,img_predited)
    i+=1
    print(i)
