# Sign Language to Text using Gesture Detection

This project uses gesture recognition techniques to translate American Sign Language (ASL) gestures into text. By leveraging computer vision and machine learning algorithms, it detects and interprets hand movements to generate corresponding text representations in real time.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview <a name="overview"></a>
The Sign Language to Text using Gesture Detection project utilizes Convolutional Neural Networks (CNN) to classify and translate sign language gestures into text. This can help bridge the communication gap between deaf and hearing individuals.

## Features <a name="features"></a>
- Real-time gesture detection and translation
- Custom dataset creation and training
- High accuracy in recognizing ASL gestures
- Easily extendable for additional gestures

## Installation <a name="installation"></a>
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

## Usage <a name="usage"></a>
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
   
## Acknowledgements <a name="acknowledgements"></a>
- [Mediapipe](https://github.com/google-ai-edge/mediapipe) for hand gesture recognition
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for model training and evaluation
- [OpenCV](https://opencv.org/) for image processing

For more details, check the project [documentation](https://github.com/Namans12/Sign-Language-to-Text-using-Gesture-Detection/tree/a38a0da9a7499b5003bf320be62c97a3a4680215/Docs).
