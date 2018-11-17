# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:40:11 2018

@author: 6000021641
"""
import sys
import os

save_dir = "./36-12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_36-12.txt'), 'r')
f2 = open(os.path.join(save_dir, 'neg_36-12.txt'), 'r')
f3 = open(os.path.join(save_dir, 'part_36-12.txt'), 'r')

pos = f1.readlines()
neg = f2.readlines()
part = f3.readlines()
f = open(os.path.join(save_dir, 'label-train.txt'), 'w')

for i in range(int(len(pos))):
    p = pos[i].find(" ") + 1
    pos[i] = pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    f.write(pos[i])

for i in range(int(len(neg))):
    p = neg[i].find(" ") + 1
    neg[i] = neg[i][:p-1] + ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1\n"
    f.write(neg[i])

for i in range(int(len(part))):
    p = part[i].find(" ") + 1
    part[i] = part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n"
    f.write(part[i])

f1.close()
f2.close()
f3.close()
