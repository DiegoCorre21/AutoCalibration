import cv2
import numpy as np

lowera = np.array([0, 63, 63])  # Red upper and lower HSV ranges
uppera = np.array([8, 255, 255])
img = cv2.imread("cube_photo8.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV transformation
mask = cv2.inRange(img_hsv, lowera, uppera)  # Mask creation
img_blur = cv2.bilateralFilter(mask, 10, 85, 85)  # Blur to decrease saw teeth in mask contours
img_canny = cv2.Canny(img_blur, 10, 50)  # Canny to find edges
img_dilation = cv2.dilate(img_canny, (5, 5), iterations=4)  # Morphological transformations
img_erode = cv2.erode(img_dilation, (5, 1), iterations=1)
img_dilation = cv2.dilate(img_erode, (7, 7), iterations=5)
img_blur2 = cv2.GaussianBlur(img_dilation, (31, 31), 5)  # Blur to further decrease saw teeth
corners = cv2.goodFeaturesToTrack(img_blur2, 4, 0.000001, 100)  # Shi-Tomasi Detector to find corners.
corners_x = np.array([])
corners_y = np.array([])
if corners is not None:
    corners = np.int0(corners)
    for corner in corners:  # Iterate through corners detected
        x, y = corner.ravel()  # Same as x, y = corner[0]
        corners_x = np.concatenate((corners_x, [x]), 0)  # Add the x coordinates to corners_x matrix
        corners_y = np.concatenate((corners_y, [y]), 0)  # Add the y coordinates to corners_y matrix
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)  # Create dots in the corners
        cv2.putText(img, f'{x}, {y}', (x - 10, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                    1, cv2.LINE_AA)  # Write coordinates of each corner
cv2.imshow("corners", img)
k = cv2.waitKey(0)
