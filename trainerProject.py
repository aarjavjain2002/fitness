import cv2
import numpy as np
import time
import poseModule as pm
import pandas as pd
import os
from accuracy import get_accuracy

cap = cv2.VideoCapture(0)   

detector = pm.poseDetector()

reps = 0
direction = 0 #- 0 means up, 1 means down -> full rep

#- Getting the start and end angles from CSV
# Define the path to the CSV file
folder_path = "trainingData"
csv_filename = "bicepCurl.csv"
csv_file_path = os.path.join(folder_path, csv_filename)

#- Read the CSV file into a DataFrame
train = pd.read_csv(csv_file_path)

#- Calculate the min and max angles from the df
startAngle = train['1'].min()
endAngle = train['1'].max()

#- Pandas dataframe here to store the angle after every frame (every 1ms) here
angles_df = pd.DataFrame(columns=['1'])

while True:
    # #- Reading the image
    success, img = cap.read()

    startTime = time.time()

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


        if angle >= startAngle:
            #- Add the angle to the DataFrame
            angles_df = pd.concat([angles_df, pd.DataFrame([angle], columns=['1'])], ignore_index=True)

        #- Percentage (going between 220 and 300 for now)
        percentage = np.interp(angle, (startAngle, endAngle), (0, 100))

        #- Rectangle bar
        bar = np.interp(angle, (startAngle, endAngle), (650, 100))

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

        if reps >= 1:
            break;
    
    endTime = time.time()

    print(endTime - startTime)

    #- Displaying the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

stepSize = len(angles_df) / 100

indices = [round(stepSize * i) for i in range(100)]

selected_rows = angles_df.iloc[indices]

pd.DataFrame({"Train" : list(train["1"]), "Actual" : list(selected_rows["1"])}).to_csv("comparison.csv")

print(get_accuracy(selected_rows, train))