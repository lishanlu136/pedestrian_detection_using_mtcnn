# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:59:38 2018

@author: 6000021641
"""

import os
import sys
import re

#用于读取标注的矩形框的位置，模式为(Xmin, Ymin) - (Xmax, Ymax)
def read_pts(s):
    l1 = s.split(' - ')
    pts=[]
    for i in range(len(l1)):
        n = l1[i][1:-1]
        p = n.split(', ')
        pts.extend(p)
    return pts
     
root = "D:/dataset/person/INRIAPerson"
annotation_dir = root + "/Train/annotations"
fs = os.listdir(annotation_dir)
with open(root+"/"+"label.txt", "w") as of:
    for i in range(len(fs)):
        print(fs[i])
        with open(annotation_dir + "/" + fs[i]) as f:    #Open the annotation.txt files
            for line in f:                               #str
                if(line[0]=="#" or line == "\n"):
                    continue
                else:
                    if "Image filename" in line:
                        img_file = line.split(':')[-1].strip()[1:-1]           
                        of.write(img_file + ' ')
                    #if "Objects with ground truth" in line:
                    #    num_obj = line.split(':')[-1].strip().split(' { ')[0]
                    #    of.write(num_obj + ' ')
                    if "Bounding box" in line:
                        #pattern = re.compile('"(.*)"')  #提取“ ”包含的内容
                        #cls = pattern.findall(line)     #list
                        points = line.split(':')[-1].strip()
                        pts = read_pts(points)          #list
                        of.write(str(pts[0]) + ' ' + str(pts[1]) \
                                 + ' ' + str(pts[2]) + ' ' + str(pts[3]) + ' ')
            of.write("\n")

#root = "D:/dataset/person/INRIAPerson"
#with open(root+"/"+"oneObj_label.txt", "w") as of:
#    with open(root+'/'+'label.txt') as f:
#        for line in f:
#            s = line.split(' ')
#            if s[1] == '1':
#                of.write(str(s[0])+' '+str(s[3])+' '+str(s[4])+' '+str(s[5])+' '+str(s[6])+'\n')
            
          
                        
            
                
            
            
    