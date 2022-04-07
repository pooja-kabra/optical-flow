import numpy as np
import cv2
import math

cap = cv2.VideoCapture('Cars_On_Highway_1.mp4')


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = (255,0,0)

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

frame_width=1920
frame_height=1080
# Create a mask image for drawing purposes
X, Y = np.mgrid[0:frame_width:25, 0:frame_height:25]
p0 = np.vstack((X.ravel(), Y.ravel())).T
p0 = np.expand_dims(p0, axis= 1).astype("float32")

frame_width=1920
frame_height=1080
out = cv2.VideoWriter('Optical_Flow.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width,frame_height))

while(1):
    

    mask = np.zeros_like(old_frame)
    
    ret,frame = cap.read()
    if ret == True:
 
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        old_gray = frame_gray.copy()
        
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
    
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            if math.sqrt((c-a)**2 + (d-b)**2)<50:
                mask = cv2.arrowedLine(mask,(c,d), (a,b), color, 2)
    
        img = cv2.add(frame,mask)
        
        cv2.imshow('frame',img)
        print(img.shape)
        out.write(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

out.release()
cv2.destroyAllWindows()
cap.release()