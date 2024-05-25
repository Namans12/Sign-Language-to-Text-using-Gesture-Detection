import cv2
import mediapipe as mp

# Initialize the pose estimation module
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define a function to process a frame
def process_frame(frame, pose):
    # Convert the frame to RGB color-space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the pose estimation module
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    return frame

# Initialize the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Process the frame
    processed_frame = process_frame(frame, pose)

    # Display the frame
    cv2.imshow('Pose Estimation', processed_frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
