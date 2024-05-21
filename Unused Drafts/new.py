import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("sign_language_model.h5")

# Capture image from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # Preprocess the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
