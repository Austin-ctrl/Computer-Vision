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
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# ... (same imports and setup code)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    calibration_angles = []
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

            # Calculate shoulder angle
            vertical_point = (right_shoulder[0], 0)
            angle = calculate_angle(left_shoulder, right_shoulder, vertical_point)

            # Calibration phase: collect 30 frames of "good posture"
            if not is_calibrated and calibration_frames < 30:
                calibration_angles.append(angle)
                calibration_frames += 1
                cv2.putText(image, f"Calibrating... {calibration_frames}/30", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif not is_calibrated:
                shoulder_threshold = np.mean(calibration_angles) - 10  # simple offset
                is_calibrated = True
                print(f"Calibration done. Threshold: {shoulder_threshold:.1f}")

            # Posture feedback
            if is_calibrated:
                if angle < shoulder_threshold:
                    color = (0, 0, 255)  # Red (bad posture)
                    status = "Bad Posture"
                else:
                    color = (0, 255, 0)  # Green (good posture)
                    status = "Good Posture"

                cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, f"Angle: {angle:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the video feed
        cv2.imshow('Posture Detection', image)

        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
