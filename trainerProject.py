import cv2
import numpy as np
import time
import poseModule as pm

# cap = cv2.VideoCapture(0)   

detector = pm.poseDetector()

while True:
    # #- Reading the image
    # success, img = cap.read()

    # #- Resizing the image
    # img = cv2.resize(img, (1280, 720))

    img = cv2.imread("assets/test.jpg")

    img = detector.findPose(img, draw=True)

    #- Displaying the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

