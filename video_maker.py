# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:33:05 2019

@author: akihiro
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4','v' )
video = cv2.VideoWriter('sampleFolder/test.mp4', fourcc, 20.0, (480, 480)) #完成させるファイル名とpathを指定
for time in range(300): #
    img = cv2.imread("sampleFolder/time="+str(int(time+1))+"_test.png")
    #print(img)
    img = cv2.resize(img, (480, 480))
    #print(img)
    video.write(img)
    
video.release()
print("Done")