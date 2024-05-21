import cv2
import mediapipe as mp

# Import the Pose module and the draw_landmarks function
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.pose import draw_landmarks

def process_frame(frame):
    # Convert the frame to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize the pose estimation module
    pose = Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Process the frame with the pose estimation module
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        draw_landmarks(frame, results.pose_landmarks, Pose.POSE_CONNECTIONS)

    return frame

# Initialize the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the frame
    cv2.imshow('Pose Estimation', processed_frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
