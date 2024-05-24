# Sign Language to Text using Gesture Detection

This project uses gesture recognition techniques to translate American Sign Language (ASL) gestures into text. By leveraging computer vision and machine learning algorithms, it detects and interprets hand movements to generate corresponding text representations in real time.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

## Overview
The Sign Language to Text using Gesture Detection project utilizes Convolutional Neural Networks (CNN) to classify and translate sign language gestures into text. This can help bridge the communication gap between deaf and hearing individuals.

## Features
- Real-time gesture detection and translation
- Custom dataset creation and training
- High accuracy in recognizing ASL gestures
- Easily extendable for additional gestures

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Namans12/Sign-Language-to-Text-using-Gesture-Detection.git
   cd Sign-Language-to-Text-using-Gesture-Detection
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements_pip.txt
   # Or for Conda users
   conda install --file requirements_conda.txt
   ```

## Usage
1. Collect data using the provided script:
   ```bash
   Copy code
   python collect-data.py
   ```
2. Train the model:
   ```bash
   Copy code
   python train.py
   ```
3. Run the real-time gesture detection:
   ```bash
   Copy code
   python Camera Feed.py
   ```

## Dataset
The dataset used for training is custom made. You can download the dataset from the following links:
- Train : https://drive.google.com/drive/u/1/folders/1-XTAjPPRPFeRqu3848z8dMXaolILWizn
- Test : https://drive.google.com/drive/u/1/folders/18e1F1n1SWPF8lUF8pCKdUzSzKAbmSbVN

## Acknowledgements
- Mediapipe for hand gesture recognition
- TensorFlow and Keras for model training and evaluation
- OpenCV for image processing

For more details, check the project [documentation](#documentation)
