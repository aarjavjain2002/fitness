# import cv2
# import numpy as np
# import time
# import poseModule as pm
# import pandas as pd
# import os

# # Initialize the pose detector
# detector = pm.poseDetector()

# # Start capturing video
# cap = cv2.VideoCapture('assets/bicepCurl.mov')

# # Get the total number of frames in the video
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Calculate the interval at which to capture frames
# interval = total_frames // 100

# #- Pandas Train values
# df = pd.DataFrame(columns=["1"], index=range(100))

# # Initialize a counter for the number of captured angles
# captured_angles = 0

# # Counter for the current frame
# current_frame = 0

# while True:
#     success, img = cap.read()

#     # Break the loop if the video ends or capturing fails
#     if not success:
#         break

#     # Resize the image
#     img = cv2.resize(img, (140, 318))

#     # Process the image and find the pose
#     img = detector.findPose(img, draw=False)
#     lmList = detector.findPosition(img, draw=False)

#     # Only capture the angle at the specified intervals
#     if current_frame % interval == 0 and len(lmList) != 0:
#         # Left arm angle
#         angle = detector.findAngle(img, 11, 13, 15)
#         df.loc[captured_angles, "1"] = angle
#         captured_angles += 1

#         # Break if we have captured 100 angles
#         if captured_angles >= 100:
#             break

#     current_frame += 1

#     # Display the image
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# #- Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()

# #- Define the path for the new directory
# folder_path = "trainingData"
# csv_filename = "angle_data.csv"
# csv_file_path = os.path.join(folder_path, csv_filename)

# #- Saving the DataFrame to a CSV file in the specified directory
# df.to_csv(csv_file_path, index=False)
# print(f"Saved angle data to {csv_file_path}")

# #- At this point, 'df' datagrame contains 100 angles captured at equal intervals
# print(df)  # To verify the result