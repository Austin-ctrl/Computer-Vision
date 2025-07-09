import cv2
import mediapipe as mp
import numpy as np
#import time
#import os

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Start video capture and pose detection
cap = cv2.VideoCapture(0)
cap.set(3, 2000)  # Width
cap.set(4, 1000)   # Height


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    calibration_shoulder_angles = []
    calibration_neck_angles = []
    calibration_frames = 0
    is_calibrated = False
    shoulder_threshold = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = (
                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])
            )
            right_shoulder = (
                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])
            )
            left_ear = (
                int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * image.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * image.shape[0])
            )
            right_ear = (
                int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * image.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * image.shape[0])
            )

            # Calculate shoulder angle
            shoulder_angle = calculate_angle(left_shoulder, right_shoulder,  (right_shoulder[0], 0))
            neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

            # Calibration phase: collect 30 frames of "good posture"
            if not is_calibrated and calibration_frames < 30:
                calibration_shoulder_angles.append(shoulder_angle)
                calibration_neck_angles.append(neck_angle)
                calibration_frames += 1
                cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            elif not is_calibrated:
                shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
                neck_threshold = np.mean(calibration_neck_angles) - 10
                is_calibrated = True
                print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")


            if is_calibrated:
                if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                    status = "Poor Posture"
                    color = (0, 0, 255)  # Red
                else:
                    status = "Good Posture"
                    color = (0, 255, 0)  # Green

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the video feed
        cv2.imshow('Posture Detection', image)

        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
