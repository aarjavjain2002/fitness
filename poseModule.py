import cv2
import mediapipe as mp
import time
import math
import pandas as pd
import numpy as np


class poseDetector():

    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initializes the pose detector.
        :param mode: Whether to treat the input images as a batch of static and possibly unrelated images, or a stream of images where each image is related to the previous one. Default is False.
        :param model_complexity: Complexity of the pose landmark model: 0, 1 or 2. Higher numbers improve accuracy but reduce speed. Default is 1.
        :param smooth_landmarks: Whether to filter landmarks across different input images to reduce jitter. Default is True.
        :param min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for the detection to be considered successful. Default is 0.5.
        :param min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the landmark-tracking to be considered successful. Default is 0.5.
        """
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.reps_count = 0
        self.reps = [pd.DataFrame()]

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # Update to the new Pose constructor call
        self.pose = self.mpPose.Pose(static_image_mode=mode, model_complexity=model_complexity, 
                                     smooth_landmarks=smooth_landmarks, min_detection_confidence=min_detection_confidence, 
                                     min_tracking_confidence=min_tracking_confidence)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        #- Declaring lmList as an object property to use it internally in the class (specifically, the findAngle method)
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                print(f"in find position. {img.shape}")
                print(f"in find position. id: {id}, lm: {lm}")

                #- Calculating the pixel value of the landmark
                #- by multiplying the normalized coordinates with the height and width of the image
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        #- Returns the the index and the coordinates of each landmark
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        #-  Get the landmark coordinates
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        #- Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (10, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
        return angle

    def append_storage(self,c_time,img):
        c_df = self.reps[self.reps_count]
        for id,lm in enumerate(self.results.pose_landmarks.landmark):
            h,w = img.shape
            # not sure how to find the z-coordinate of the lm so this is a placeholder
            c_df.add(np.array([c_time,int(lm.x *w),int(lm.y * h), lm.z]))




def main():
    cap = cv2.VideoCapture('PoseVideos/1.mp4')
    pTime = 0
    detector = poseDetector()
    start_time = time.time()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        c_time = cTime - start_time
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()