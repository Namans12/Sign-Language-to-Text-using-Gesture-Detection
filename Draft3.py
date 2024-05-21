import cv2
import mediapipe as mp

# Load the Mediapipe Pose model
mp_pose = mp.solutions.pose

# Initialize the pose detection module
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # Read an image
    image = cv2.imread("image.jpg")

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Set the image data for detection
    pose_results = pose.process(image)

    # Print the pose detection results
    print(pose_results)
