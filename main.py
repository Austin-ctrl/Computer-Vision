import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Intializing MediaPipe Pose and webcam
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0])
    angle = np.abs(radians * 180./np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(0)
    cap.set(3, 1800)
    cap.set(4, 900)
    cap.set(10, 100)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
            right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
            right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0]))
            shoulder_angle = calculate_angle(left_shoulder, right_shoulder, right_hip)
            
            left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
            nose = (int(landmarks[mp_pose.PoseLandmark.NOSE].x * image.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.NOSE].y * image.shape[0]))
            neck_angle = calculate_angle(left_ear, left_shoulder, nose)

            shoulder_threshold = 35
            neck_threshold = 40

            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                posture_status = "Bad Boy"
                color = (0, 0, 255)
            else:
                posture_status = "Good Boy"
                color = (0, 255, 0)

            cv2.putText(image, posture_status, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)            
            cv2.putText(image, "Shoulder Angle:" + str(shoulder_angle), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "Neck Angle:" + str(neck_angle), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)

        except:
            pass
            
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

