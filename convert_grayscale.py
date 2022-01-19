"""
Process the depth npy files and resize them into training datas
(Still have problem when the hand is too close to the wall)
@author: Shen Jie Koh
"""

import os
import numpy as np
import cv2

path = "../data/raw/depth_16bit" #../ previous directory
save_path = "../data/train"
depth_numpy = os.listdir(path)  # get allfilenames in path
depth_scale = 0.00025
min_dist = 0.25
max_dist = 0.5
min_grayvalue = 55
max_grayvalue = 255

for data in depth_numpy:
    data_path = os.path.join(path, data)
    depth_data = np.load(data_path)
    distance = depth_data*depth_scale
    distance[(distance<min_dist)|(distance>max_dist)] = 0.0 # set data out of range as 0
    gray_npy = []
    resized = []
    
    for i in range(0, depth_data.shape[0]):
        min_value = np.min(distance[i][np.nonzero(distance[i])])
        max_value = min_value + 0.15    # 15 cm behind the fingertip
        grayscale = np.interp(distance[i],[min_value,max_value],
                              [max_grayvalue,min_grayvalue],left=0,right=0)
        grayscale = grayscale.astype('uint8')
        gray_npy.append(np.expand_dims(grayscale,0))
    gray_npy = np.concatenate(gray_npy,axis=0)
    gray_npy2 = np.copy(gray_npy)
    
    # Comparasion between BackgroundSubtactorKNN result and manually cropped result
    bs = cv2.createBackgroundSubtractorKNN()
    mask = np.zeros(gray_npy.shape,dtype='uint8')
    for i in range(gray_npy.shape[0]):
        ret, thresh = cv2.threshold(gray_npy[i], 10, 255, cv2.THRESH_BINARY)
        vid_mask = bs.apply(gray_npy2[i])
        contours1, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(vid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour1 = contours1[0]
        for cnt in contours1:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > cv2.contourArea(max_contour1) and not (w >= 320 or h >= 240):
                max_contour1 = cnt
        (max_x1, max_y1, max_w1, max_h1) = cv2.boundingRect(max_contour1) #手部contour位置參數
        
        max_contour2 = contours2[0]
        for cnt in contours2:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > cv2.contourArea(max_contour2) and not (w >= 320 or h >= 240):
                max_contour2 = cnt
        (max_x2, max_y2, max_w2, max_h2) = cv2.boundingRect(max_contour2) #手部contour位置參數
        #print((i, max_x, max_y, max_w, max_h, cv2.contourArea(max_contour)))
        
        if cv2.contourArea(max_contour1) > cv2.contourArea(max_contour2):
            mask[i][max_y1:max_y1+max_h1,max_x1:max_x1+max_w1] = 1
            gray_npy2[i] = gray_npy2[i] * mask[i]
        else:
            mask[i][max_y2:max_y2+max_h2,max_x2:max_x2+max_w2] = 1
            gray_npy2[i] = gray_npy2[i] * mask[i]
        if not gray_npy2[i].any():
            gray_npy2[i] = gray_npy2[i-1]
            
        img = cv2.resize(gray_npy2[i], (224, 168), interpolation=cv2.INTER_AREA)
        img = np.pad(img, pad_width=[(28, 28),(0, 0)], mode='constant')
        img = np.expand_dims(np.copy(img), axis=0)
        resized.append(img)
    resized_grayscale = np.concatenate(resized, axis=0)
    np.save(os.path.join(save_path,data), resized_grayscale)
    print(os.path.join(save_path,data))
