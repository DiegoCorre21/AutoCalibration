import cv2
from Funciones import *

frame = cv2.imread("cube_photo8.jpg")
create_trackbars()
while True:
    mask = trackbar(frame, "track")
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
