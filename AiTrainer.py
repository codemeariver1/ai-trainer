import cv2
import numpy as np
import time
import PoseModule as pm

# Capture live feed or video
capture = cv2.VideoCapture(0)

# Initialize the pose detector
detector = pm.PoseDetector()

arm_rep = 0
arm_dir = 0 # 0:open, 1:close
prev_time = 0

while True:
    # Don't need if capturing image
    success, img = capture.read()
    # img = cv2.resize(img, (1280, 720))

    # Capture image
    # img = cv2.imread("AiTrainer/")

    # Set the pose finder on the image
    img = detector.findPose(img, False)

    # Get the landmark values
    landmark_list = detector.findPosition(img, False)
    #print(landmark_list)
    if len(landmark_list) != 0:
        # Left arm, distal direction
        arm_angle = detector.findAngle(img, 11, 13, 15)
        # Right arm, distal direction
        # arm_angle = detector.findAngle(img, 12, 14, 16)

        # Left side view percentage
        perc = np.interp(arm_angle, (200, 310), (0, 100))
        # Right side view percentage
        # perc = np.interp(arm_angle, (50, 150), (100, 0))

        bar = np.interp(arm_angle, (220, 310), (650, 100))

        print(arm_angle, perc)

        # Check for the rep
        color = (11, 190, 255)
        if perc == 100:
            color = (0, 255, 0)
            if arm_dir == 0:
                arm_rep += 0.5
                arm_dir = 1
        if perc == 0:
            color = (0, 0, 255)
            if arm_dir == 1:
                arm_rep += 0.5
                arm_dir = 0
        #print(arm_rep)

        # Draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(perc)}', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        # Draw repetition counter
        cv2.rectangle(img, (0, 540), (320, 690), (66, 45, 43), cv2.FILLED)
        cv2.putText(img, str(int(arm_rep)), (45, 670), cv2.FONT_HERSHEY_PLAIN,  10, (60, 35, 239), 25)

    # Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (40, 255, 10), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)