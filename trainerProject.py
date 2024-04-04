import cv2
import numpy as np
import time
import poseModule as pm

cap = cv2.VideoCapture(1)   

detector = pm.poseDetector()

reps = 0
direction = 0 #- 0 means up, 1 means down -> full rep

while True:
    # #- Reading the image
    success, img = cap.read()

    # #- Resizing the image
    img = cv2.resize(img, (1280, 720))

    # img = cv2.imread("assets/test.jpg")

    #- Test image
    img = detector.findPose(img, draw = False) #- The false is to remove the other lines and only focus on the landmarks we want

    #- Getting the landmarks
    lmList = detector.findPosition(img, draw = False)
    
    if len(lmList) != 0:
        # #- Right arm
        # detector.findAngle(img, 12, 14, 16)

        #- Left arm
        angle = detector.findAngle(img, 11, 13, 15)

        #- Percentage (going between 220 and 300 for now)
        percentage = np.interp(angle, (220, 300), (0, 100))

        #- Rectangle bar
        bar = np.interp(angle, (220, 300), (650, 100))

        #- Counting reps
        color = (255, 0, 255)
        if percentage == 100:
            color = (0, 255, 0)
            if direction == 0:
                reps += 0.5
                direction = 1
        if percentage == 0:
            color = (0, 255, 0)
            if direction == 1:
                reps += 0.5
                direction = 0
        
        #- Rectangle display
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(percentage)} %', (1100, 75), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)

        #- Reps display
        cv2.putText(img, "Reps: " + str(int(reps)), (30, 670), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)

    #- Displaying the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

