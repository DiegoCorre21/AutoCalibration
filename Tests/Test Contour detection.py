import cv2
import numpy as np

mask = cv2.imread("mask.png")  # Insert the path for the mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale image
frame = cv2.imread("frame.png")  # Insert the path for the frame
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#  Find the contours in the mask binary image
if len(contours) != 0:  # If contours are detected
    for cnt in contours:  # Iterate through all closed contours
        area = cv2.contourArea(cnt)
        if area > 1000:  # If the area of the contours are big enough
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, peri * 0.02, True)  # Approximate a polygon to enclose the contour
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 2)  # Draw contours in the original frame
            x, y, w, h = cv2.boundingRect(cnt)  # Create bounding rectangle
            cx, cy = x + (w // 2), y + (h // 2)  # define center coordinates of the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # Draw dot at center
            cv2.putText(frame, f'{cx}, {cy}', (cx + 20, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (108, 125, 24),
                        1, cv2.LINE_AA)  # Write the coordinates on the frame
cv2.imshow("mask", mask)
cv2.imshow("frame", frame)
cv2.waitKey(0)
