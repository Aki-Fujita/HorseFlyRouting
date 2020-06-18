# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:33:05 2019

@author: akihiro
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4','v' )
video = cv2.VideoWriter('horseFlyMitou.mp4', fourcc, 20.0, (640, 480))
for time in range(2000):
    
    if time >= 0 and time <= 389:
        img = cv2.imread("forMovieMultiple/time="+str(int(time*2))+"_test.png")
        #print(img)
        img = cv2.resize(img, (640, 480))
        #print(img)
        video.write(img)
        
video.release()
print("Done")