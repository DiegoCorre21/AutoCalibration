import realsense_depth
import cv2
import threading
import socket
import time
from Funciones import *
import re


#  For gripper use
close = 1
open = 0
# Set parameters
data = np.load("Your path to the hand-camera camera calibration npz file")
camMatrix_hand = data["camMatrix"]
distortion_hand = data["distCoef"]
data = np.load("Your path to the fixed-camera camera calibration npz file")
camMatrix_fixed = data["camMatrix"]
distortion_fixed = data["distCoef"]
data = np.load("Your path to the hand-eye calibration matrix npz file")
hg2c = data["h_c2g"]
TCP_IP = '192.168.0.1' #  Robot IP address. Start the TCP server from the robot before starting this code
TCP_PORT = 3000  #  Robot Port
BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096
home = [350, 150, 150]  #  Starting position of the robot
maskSum = np.zeros((480, 640), dtype=np.uint8)  #  Initializing a mask
object_points = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [50, 50, 0]], dtype=np.float32)  #  Object corner points array
#  The 4 points are the left-upper corner, the right-upper corner, the left-bottom corner, and the right-bottom corner respectively
global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server. 
                                                       #  Connect the ethernet cable to the robot electric box first
c.connect((TCP_IP, TCP_PORT))  

cap_fixed = cv2.VideoCapture(1, cv2.CAP_DSHOW)  #  Initialize the capture of frames from the Logitech C920 fixed camera
cap_fixed.set(3, 640)  #  Set to 640x480
cap_fixed.set(4, 480)
cap_hand = realsense_depth.DepthCamera()  #  Initialize the capture of frames from the IntelRealSense D435i camera

state = 0  # 0 -> Fixed camera use, 1 -> Hand camera use
cube_target = 0  # 0 -> Red, 1 -> Yellow, 2 -> Green, 3 -> None
position = home  #  Initialize position of the robot as home
robot_move(350, 150, 150, 180, 0, 90, c)  #  Move the robot to the home position.
rPos = receive(c)  #  Receive the position of the robot after moving
nstack = 0  #  Amount of stacked cubes starts at 0
# ------------------------------------------------------------------------------
while True:
    # Get Robot position
    if state == 0:
        # RGB Image acquisition
        for i in range(1, 5):
            _, frame_fixed = cap_fixed.read()

        # ------------------------------------------------------------------------------

        # Color Detection & Centerpoint detection
        red, yellow, green = detect_cubes(frame_fixed, camMatrix_fixed, distortion_fixed)

        # ------------------------------------------------------------------------------
        cube_target = 3
        if green is not None:
            cube_target = 2
            position = green
        if yellow is not None:
            cube_target = 1
            position = yellow
        if red is not None:
            cube_target = 0
            position = red
        # Send robot to cube position
        if cube_target != 3:
            robot_move(position[0]+30, position[1]-60, 70, 180, 0, 90, c)
            rPos = receive(c)
            position = home
            state = 1  #  Set the state to 1, meaning the fixed camera turns off, and the hand camera turns on

        # ------------------------------------------------------------------------------

    if state == 1:
        gripMove(open)  #  Open the gripper
        hg2o = get_robot_pos(rPos)  #  Use the position received from the robot to calculate the 4x4 homogeneous transformation from the gripper 
                                    #  to the robot world coordinates.
        # RGBD Image Acquisition
        if cube_target == 0:  #  Check which color was detected to set the mask to detect that color
            for i in range(1, 20):
                _, _, frame_hand = cap_hand.get_frame()
                mask = trackbar(frame_hand, "red_hand")
                maskSum = cv2.bitwise_or(maskSum, mask)


        if cube_target == 1:
            for i in range(1, 20):
                _, _, frame_hand = cap_hand.get_frame()
                mask = trackbar(frame_hand, "yellow_hand")
                maskSum = cv2.bitwise_or(maskSum, mask)


        if cube_target == 2:
            for i in range(1, 20):
                _, _, frame_hand = cap_hand.get_frame()
                mask = trackbar(frame_hand, "green_hand")
                maskSum = cv2.bitwise_or(maskSum, mask)

        # ------------------------------------------------------------------------------

        # Corner Detection
        corners = get_corners(frame_hand, mask)  #  Process the mask and use the resulting image to find the corners using the 
                                                 #  Shi-Tomasi approach called Corners Good Features
        # ------------------------------------------------------------------------------

        # Pose estimation with PNP
        if corners is not None:  #  If corners are detected, estimate the pose of the target cube
            ht2o, centre = pnpSolve(frame_hand, corners, camMatrix_hand, distortion_hand, hg2o, hg2c) 
            grasp(centre, ht2o, rPos, c)  #  Move the robot to grasp the cube
            stackMove(nstack, c)  #  With the cube in hand, move it to the stacking area
            nstack = nstack + 1  #  Add 1 to the amount of stacked cubes 
            state = 0  #  Return to using the fixed camera, stop using the hand camera
        else:  #  If no corners are detected, return to the initial position and use the fixed camera
            state = 0
            robot_move(350, 150, 150, 180, 0, 90, c)
            rPos = receive(c)
          
        maskSum = np.zeros((480, 640), dtype=np.uint8)  #  Set the masks used to 0 again
        
        # ------------------------------------------------------------------------------

    key = cv2.waitKey(1)  #  If q is pressed, stop the code
    if key == ord("q"):
        break
