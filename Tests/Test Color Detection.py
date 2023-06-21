import cv2
import numpy as np


lowera = np.array([5, 90, 100])  # Establish the lower and upper HSV values
uppera = np.array([50, 255, 255])
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
while True:
    _, frame = cap.read()
    HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Transform to HSV coordinates
    mask = cv2.inRange(HSV_frame, lowera, uppera)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", mask)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.imwrite("mask.png", mask)
        break

