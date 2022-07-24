import cv2
from HandTrackingModule import HandDetector
from ClassificationModule import Classifier
import numpy as np
import math
import time

wCam, hCam = 1080, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# folder = "Data/C"
# counter = 0

message = ""
startProcessingTime = 0
processingLabel = ""
processedLabel = ""
alreadyProcessed = False

# labels = ["A", "L", "SP", "NO"]
labels = ["1", "3", "4", "L", "I", "N", "H", "A", "P", "NO", "SP"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    try:
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
                prediction, index = classifier.getPrediction(imgWhite, draw=True)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=True)
                print(prediction, index)

            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)

            # xu ly text hien thi
            if(labels[index] != processingLabel):
                processingLabel = labels[index]
                startProcessingTime = time.time()
                alreadyProcessed = False
            else:
                if alreadyProcessed:
                    message += ""
                else:
                    if((time.time()-startProcessingTime) > 1.5):
                        if(processingLabel != "NO"):
                            if(processingLabel == "SP"):
                                message += " "
                                # print("Space done")
                            else:
                                message += processingLabel
                            alreadyProcessed = True

            cv2.rectangle(imgOutput, (0, 0), (1500, 150), (220, 220, 220), cv2.FILLED)
            cv2.putText(imgOutput, message, (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, 
                            (0, 0, 0), 6, cv2.LINE_AA, False)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            message = ""
    except:
        print("out of bound")
        message = ""

    cv2.imshow("Image", imgOutput)
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break