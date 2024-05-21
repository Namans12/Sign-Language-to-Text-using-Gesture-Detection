import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load an image
image = cv2.imread('example.jpg')

# Detect pose landmarks
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    results = pose.process(image)

    # Draw landmarks on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the annotated image
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
