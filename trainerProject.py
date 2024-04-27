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

# # Store the angles
# bicepTrainAngles = [183.90145456508625, 184.17043652484213, 187.30525327504256, 190.44781754947394, 195.16673345961004, 198.68618649510543, 205.24327480975057, 206.60839139537237, 217.40535663140858, 231.48307369289728, 235.873520395255, 241.84931141182506, 242.0016175231156, 248.48210682912455, 251.50939720768184, 257.5707230338824, 263.6789595125105, 264.8978347476418, 267.64321682281184, 267.6440166806398, 268.44442665161614, 271.2188752351313, 272.5143670302412, 277.20095504467895, 281.509221338209, 283.2564137022956, 289.2482304453457, 288.8689994550614, 291.4825411943545, 292.3543166286264, 292.23703304044784, 294.3994359241751, 294.8505744709624, 294.58157448256867, 297.13148678587874, 302.0053832080835, 308.4019684195557, 305.1647805509236, 310.0932631826215, 313.99491399474584, 315.0, 315.97831644817086, 316.41206836452034, 316.92502366379927, 317.66763638101924, 318.149721988367, 317.66763638101924, 317.66763638101924, 316.0329603788824, 313.0204154834996, 313.97899518442733, 312.3933577733304, 312.3933577733304, 311.32550047917033, 308.5251849597405, 309.3269778600747, 309.78385286854814, 309.2072035049678, 306.414065688195, 305.0625163543267, 302.33246186096085, 297.1103506315811, 295.43436022584, 292.8680904962846, 287.3244838104447, 281.54284086394074, 275.0445299910328, 275.3608940559684, 272.40325509524473, 265.7892187383866, 264.87415391430824, 264.0813883717865, 264.0813883717865, 263.0887728809753, 257.88229094951413, 256.15302930115047, 249.67927106188904, 249.22395766809436, 247.42903767796963, 244.0001803286019, 242.35402463626133, 238.17636877549688, 237.56608316962732, 233.54429246474876, 228.11186362800396, 222.75927353446644, 219.55097405683932, 217.94799511780798, 209.36789020238587, 204.53081036751786, 202.55087548166696, 201.00271778268902, 196.35750851430737, 196.26838339350562, 192.09341566387084, 189.84550069288883, 186.52150689860892, 184.11592665874497, 182.8954406268549, 181.66669973584317]

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

# #TODO
# okay that works perfectly. i added all this code to a separate filled called captureTrainingData. now in the main file that i am working with, i want to do something slightly different. what i want to do is capture the angles in a similar fashion as i just did for the training data. now, i will be uploading a video of me doing a curl. this time the capturing of the angles is slightly different. i want to capture 100 angles once again. but 