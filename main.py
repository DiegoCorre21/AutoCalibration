import realsense_depth
import cv2
import threading
import socket
import time
from Funciones import *
import re



close = 1
open = 0
# Set parameters
data = np.load("A:/_UPTP-Gonzalo_Miltos/2022-UPTP_Gonzalo_Miltos/Opencv/clock/venv/main/"
               "calib_data/MultiMatrix_3D_640_480.npz")
camMatrix_hand = data["camMatrix"]
distortion_hand = data["distCoef"]
data = np.load("A:/_UPTP-Gonzalo_Miltos/2022-UPTP_Gonzalo_Miltos/Opencv/clock/venv/main/"
               "calib_data/MultiMatrix_fixed_640_480.npz")
camMatrix_fixed = data["camMatrix"]
distortion_fixed = data["distCoef"]
data = np.load("A:/_UPTP-Gonzalo_Miltos/2022-UPTP_Gonzalo_Miltos/Opencv/clock/venv/main/calib_data/H_cam2grip.npz")
hg2c = data["h_c2g"]
TCP_IP = '192.168.0.1'
TCP_PORT = 3000
BUFFER_SIZE = 1024
home = [350, 150, 150]
maskSum = np.zeros((480, 640), dtype=np.uint8)
object_points = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [50, 50, 0]], dtype=np.float32)

global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect((TCP_IP, TCP_PORT))

cap_fixed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap_fixed.set(3, 640)
cap_fixed.set(4, 480)
cap_hand = realsense_depth.DepthCamera()

state = 0  # 0 -> Fixed camera use, 1 -> Hand camera use
cube_target = 0  # 0 -> Red, 1 -> Yellow, 2 -> Green, 3 -> None
position = home
robot_move(350, 150, 150, 180, 0, 90, c)
rPos = receive(c)
nstack = 0
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
        # Send to Robot
        if cube_target != 3:
            robot_move(position[0]+30, position[1]-60, 70, 180, 0, 90, c)
            state = 1
            rPos = receive(c)
            position = home

        # ------------------------------------------------------------------------------

    if state == 1:
        gripMove(open)
        hg2o = get_robot_pos(rPos)
        # RGBD Image Acquisition
        if cube_target == 0:
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
        corners = get_corners(frame_hand, mask)
        # ------------------------------------------------------------------------------

        # Pose estimation with PNP
        if corners is not None:
            ht2o, centre = pnpSolve(frame_hand, corners, camMatrix_hand, distortion_hand, hg2o, hg2c)
            
            
            grasp(centre, ht2o, rPos, c)
            
            stackMove(nstack, c)
            nstack = nstack+1
            state = 0
        else:
            state = 0
            robot_move(350, 150, 150, 180, 0, 90, c)
            rPos = receive(c)
        # ------------------------------------------------------------------------------

        # Drop to z=40*n mm

        # ------------------------------------------------------------------------------

        # Robot Return to origin
        maskSum = np.zeros((480, 640), dtype=np.uint8)
        
        # ------------------------------------------------------------------------------

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
