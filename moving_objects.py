# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:08:33 2021

@author: pooja
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('Cars_On_Highway.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_width=1920
frame_height=1080
out = cv2.VideoWriter('Moving_objects.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width,frame_height))

while(1):
    ret, frame = cap.read()
    if ret==True:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        img2_fg = cv2.bitwise_and(frame, fgmask)
    
        cv2.imshow('frame',img2_fg)
        out.write(img2_fg)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
out.release()
cap.release()
cv2.destroyAllWindows()