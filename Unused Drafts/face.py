import cv2
import numpy as np
import os

# Define the lower and upper boundaries of the skin color in HSV
skin_lower = np.array([0, 20, 70], dtype=np.uint8)
skin_upper = np.array([20, 255, 255], dtype=np.uint8)

# Create a window to display the camera feed
cv2.namedWindow("Camera Feed")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a folder to store the collected gesture data
gesture_name = input("Enter gesture name: ")
data_path = "gesture_data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)
gesture_path = os.path.join(data_path, gesture_name)
if not os.path.exists(gesture_path):
    os.makedirs(gesture_path)

# Initialize variables for data collection
collect_data = False
count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask based on the skin color boundaries
    mask = cv2.inRange(hsv, skin_lower, skin_upper)

    # Apply a series of morphological transformations to the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find the contours of the hand in the masked image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Check if data collection is enabled and save the image if so
    if collect_data:
        count += 1
        file_name = f"{gesture_name}_{count}.jpg"
        file_path = os.path.join(gesture_path, file_name)
        cv2.imwrite(file_path, frame)
        print(f"Saved {file_path}")

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Start or stop data collection if the 'c' key is pressed
    if key == ord('c'):
        collect_data = not collect_data
        if collect_data:
            print("Started data collection")
        else:
            print("Stopped data collection")
            count = 0

    # Exit the program if the 'q' key is pressed
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
