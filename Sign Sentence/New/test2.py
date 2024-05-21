import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf
from keytotext import labels, dict, nlp

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

#folder = "Data/F"
counter = 0

string_counts = {}

ch = 0
cnt = 0
sentence = ""
fsentence = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if labels[index] in string_counts:
            # If it is, increment the count by 1
            string_counts[labels[index]] += 1
            if(string_counts[labels[index]] == 10):
                sentence += labels[index]
                sentence += " "
                if(dict.get(sentence) != None):
                    fsentence = nlp(sentence)
                cnt = 0
                string_counts = {}
        else:
            # If it's not, add it to the dictionary with a count of 1
            string_counts[labels[index]] = 1

        # if(ch == index):
        #     cnt += 1
        # else:
        #     ch = index
        #     cnt = 0

        # if(cnt == 15):
        #     sentence += labels[index]
        #     sentence += " "
        #     if(dict.get(sentence) != None):
        #         fsentence = dict[sentence]
        #     cnt = 0

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.putText(imgOutput, sentence, (10, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(imgOutput, fsentence, (10, 300),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", imgOutput)
    # cv2.waitKey(1)
    key = cv2.waitKey(100)

    if(key == ord("r")):
        ch = 0
        cnt = 0
        sentence = ""
        fsentence = ""
        string_counts = {}
