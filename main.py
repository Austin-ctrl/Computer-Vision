import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Intializing MediaPipe Pose and webcam
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,  # Tells MediaPipe whether you're processing individual static images
    min_detection_confidence=0.5,  # Sets the threshold for how confident the model must be before it accepts a pose detection.
    min_tracking_confidence=0.5  # Sets how confident the model must be in tracking a pose between frames
)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmarks

    cv2.imshow('Mediapipe Feed', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
       break
# monkey ah add

cap.release()
cv2.destroyAllWindows()