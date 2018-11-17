# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:01:52 2018

@author: 6000021641
"""
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

imgH = 72
imgW = 24
#anno_file = "label.txt"
anno_file = "annotation.txt"
img_root = "D:/dataset/person" 
pos_save_dir = str(imgH) + "-" + str(imgW) + "/positive"
part_save_dir = str(imgH) + "-" + str(imgW) + "/part"
neg_save_dir = str(imgH) + "-" + str(imgW) + '/negative'
save_dir = "./" + str(imgH) + "-" + str(imgW)

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_' + str(imgH) + "-" + str(imgW) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(imgH) + "-" + str(imgW) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(imgH) + "-" + str(imgW) + '.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print ("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0
####################
neg_idx = 0

###----------利用背景图片制作负样本--------------
#neg_file = "Train/neg.lst"
#with open(neg_file, 'r') as f:
#    neg_paths = f.readlines()
#neg_len = len(neg_paths)
#print ("Have %d neg pics in total" % neg_len)
#
#for neg_path in neg_paths:
#    neg_img = cv2.imread(neg_path.strip())
#    neg_idx += 1
#    if neg_idx % 100 == 0:
#        print (neg_idx, "image done")
#    height, width, channel = neg_img.shape
#    
#    neg_num = 0
#    while neg_num < 50:
#        size = npr.randint(20, min(width, height) / 4)
#        nx = npr.randint(0, width - size)
#        ny = npr.randint(0, height - 3*size)
#        cropped_img = neg_img[ny:ny+3*size, nx:nx+size, :]
#        resized_img = cv2.resize(cropped_img, (imgW,imgH), interpolation=cv2.INTER_LINEAR)
#        save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
#        f2.write(str(imgH) + "-" + str(imgW) +"/negative/%s"%n_idx + ' 0\n')
#        cv2.imwrite(save_file, resized_img)
#        n_idx += 1
#        neg_num += 1
############################
    
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = annotation[1:]
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(img_root + '/' + im_path)
    idx += 1
    if idx % 100 ==0:
        print (idx, "images done")
    height, width, channel = img.shape
    
    neg_num = 0
    while neg_num < 100:
        size = npr.randint(20, min(width, height) / 4)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - 3*size)
        crop_box = np.array([nx, ny, nx + size, ny + 3*size])
        Iou = IoU(crop_box, boxes)
        cropped_img = img[ny:ny+3*size, nx:nx+size, :]
        resized_img = cv2.resize(cropped_img, (imgW,imgH), interpolation=cv2.INTER_LINEAR)
        
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(str(imgH) + "-" + str(imgW) +"/negative/%s"%n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_img)
            n_idx += 1
            neg_num += 1
    
    for box in boxes:
        #box(x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if w < 12 or x1 < 0 or y1 < 0:
            continue
        i = 0
        # generate positive examples and part faces
        while(i<50):
            size = npr.randint(int(w * 0.8), np.ceil(1.25 * w))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - (h/w)*size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + (h/w)*size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float((h/w)*size)
            offset_x2 = (x2 - nx1) / float(size)
            offset_y2 = (y2 - ny1) / float((h/w)*size)

            cropped_img = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_img = cv2.resize(cropped_img, (imgW,imgH), interpolation=cv2.INTER_LINEAR)
            i += 1
            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                f1.write(str(imgH) + "-" + str(imgW)+"/positive/%s"%p_idx + ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_img)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                f3.write(str(imgH) + "-" + str(imgW)+"/part/%s"%d_idx + ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_img)
                d_idx += 1

        box_idx += 1
        print ("%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()

    