import cv2
import time
import mediapipe as mp
import numpy as np

import os
os.environ["BEEWARE_ANDROID"] = "1"


def main():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Failed to open the camera.")
        exit()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          model_complexity=1,
                          min_detection_confidence=0.75,
                          min_tracking_confidence=0.75,
                          max_num_hands=2)

    mpDraw = mp.solutions.drawing_utils

    prevTime = 0
    currTime = 0

    left_hands = []
    right_hands = []

    while True:
        success, img = video.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        left_hands.clear()
        right_hands.clear()

        if results.multi_hand_landmarks:
            for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = "Left" if handedness.classification[0].label == "Right" else "Right"

                if hand_label == "Left":
                    left_hands.append(handLms)
                else:
                    right_hands.append(handLms)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    centerx, centery = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        cv2.putText(img, f"{hand_label} Hand", (centerx, centery), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)

                    # Access x, y, and z coordinates of each landmark
                    x_coordinate = lm.x
                    y_coordinate = lm.y
                    z_coordinate = lm.z
                    print(f"Hand: {hand_label}, LM{id}: ({x_coordinate:.2f}, {y_coordinate:.2f}, {z_coordinate:.2f})")

        for handLms in left_hands:
            # Palm direction detection for left hand
            hand_label = "Left"

            # Palm direction detection
            palm = mpHands.HandLandmark.WRIST

            palm_x = int(handLms.landmark[palm].x * img.shape[1])
            palm_y = int(handLms.landmark[palm].y * img.shape[0])

            thumb_tip = mpHands.HandLandmark.THUMB_TIP
            thumb_tip_x = int(handLms.landmark[thumb_tip].x * img.shape[1])
            thumb_tip_y = int(handLms.landmark[thumb_tip].y * img.shape[0])

            index_tip = mpHands.HandLandmark.INDEX_FINGER_TIP
            index_tip_x = int(handLms.landmark[index_tip].x * img.shape[1])
            index_tip_y = int(handLms.landmark[index_tip].y * img.shape[0])

            # Calculate palm direction based on the relative positions of thumb tip and index finger tip
            palm_direction = "Unknown"
            if thumb_tip_x < index_tip_x:
                palm_direction = "Forward"
            elif thumb_tip_x > index_tip_x:
                palm_direction = "Backward"

            cv2.putText(img, f"{hand_label} Palm: {palm_direction}", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 1)
            cv2.circle(img, (palm_x, palm_y), 5, (255, 0, 0), cv2.FILLED)

        for handLms in right_hands:
            # Palm direction detection for right hand
            hand_label = "Right"

            # Palm direction detection
            palm = mpHands.HandLandmark.WRIST

            palm_x = int(handLms.landmark[palm].x * img.shape[1])
            palm_y = int(handLms.landmark[palm].y * img.shape[0])

            thumb_tip = mpHands.HandLandmark.THUMB_TIP
            thumb_tip_x = int(handLms.landmark[thumb_tip].x * img.shape[1])
            thumb_tip_y = int(handLms.landmark[thumb_tip].y * img.shape[0])

            index_tip = mpHands.HandLandmark.INDEX_FINGER_TIP
            index_tip_x = int(handLms.landmark[index_tip].x * img.shape[1])
            index_tip_y = int(handLms.landmark[index_tip].y * img.shape[0])

            # Calculate palm direction based on the relative positions of thumb tip and index finger tip
            palm_direction = "Unknown"
            if thumb_tip_x < index_tip_x:
                palm_direction = "Forward"
            elif thumb_tip_x > index_tip_x:
                palm_direction = "Backward"

            if palm_direction == "Backward":
                temp_detection = "Forward"
                cv2.putText(img, f"{hand_label} Palm: {temp_detection}", (10,100), cv2.FONT_HERSHEY_COMPLEX,0.7, (0,255,0), 1)
                cv2.circle(img, (palm_x, palm_y), 5, (255, 0, 0), cv2.FILLED)
            elif palm_direction == "Forward":
                temp_detection = "Backward"
                cv2.putText(img, f"{hand_label} Palm: {temp_detection}", (10,100), cv2.FONT_HERSHEY_COMPLEX,0.7, (0,255,0), 1)
                cv2.circle(img, (palm_x, palm_y), 5, (255, 0, 0), cv2.FILLED)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live", img)

        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
