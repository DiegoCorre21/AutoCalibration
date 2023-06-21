import cv2

camera_number = 1
cap = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
#  Initialize camera communication
while True:
    _, frame = cap.read()  # Read frame from camera
    cv2.imshow("Fixed-camera image", frame)  # Display the frame
    key = cv2.waitKey(1)  # Wait for a key to be pressed for 1 ms
    if key == ord("q"):  # if "q" is pressed, break the loop
        break
