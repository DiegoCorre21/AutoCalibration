import cv2
import numpy as np
import math
import socket
import serial
import time
import binascii
import re
import math as m


def receive(c):
    data = c.recv(1024).decode()
    data = re.sub(',', ' ', data)
    pos = data.split()
    pos = list(map(float, pos))  # [x, y, z, rx - psi, ry - theta, rz - phi]
    return pos


def nothing(a):
    pass


def get_robot_pos(pos):
    Rr_e = np.array(roll_pitch_yaw(pos[5] * np.pi / 180, pos[4] * np.pi / 180, pos[3] * np.pi / 180))
    Pr_e = np.array([[pos[0], pos[1], pos[2]]])
    dummy = np.array([[0, 0, 0, 1]])
    Hr_e = np.concatenate((Rr_e, Pr_e.T), axis=1)
    Hr_e = np.concatenate((Hr_e, dummy), axis=0)
    return Hr_e


def roll_pitch_yaw(ph, t, ps):
    R_rpy = [[math.cos(ph) * math.cos(t), (-math.sin(ph) * math.cos(ps)) + (math.cos(ph) * math.sin(t) * math.sin(ps)),
              (math.sin(ph) * math.sin(ps)) + (math.cos(ph) * math.sin(t) * math.cos(ps))],
             [math.sin(ph) * math.cos(t), (math.cos(ph) * math.cos(ps)) + (math.sin(ph) * math.sin(t) * math.sin(ps)),
              (-math.cos(ph) * math.sin(ps)) + (math.sin(ph) * math.sin(t) * math.cos(ps))],
             [-math.sin(t), math.cos(t) * math.sin(ps), math.cos(t) * math.cos(ps)]]
    R_rpy = np.matrix(R_rpy).round(2)
    return R_rpy


def corners_good_features(img_corners, img):
    corners = cv2.goodFeaturesToTrack(img_corners, 4, 0.000001, 180)
    corners_x = np.array([])
    corners_y = np.array([])
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()  # Same as x, y = corner[0]
            corners_x = np.concatenate((corners_x, [x]), 0)
            corners_y = np.concatenate((corners_y, [y]), 0)
            cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(img, f'{x}, {y}', (x - 10, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                        1, cv2.LINE_AA)
    return corners_x, corners_y


def robot_move(x, y, z, rx, ry, rz, c):
    mesg = f'{x},{y},{z},{rx},{ry},{rz}'
    c.send(bytes(mesg, "utf-8"))


def get_corners(image, mask):
    img2_corners = image
    img_blur = cv2.bilateralFilter(mask, 10, 85, 85)
    img_canny = cv2.Canny(img_blur, 10, 50)
    img_dilation = cv2.dilate(img_canny, (5, 5), iterations=4)
    img_erode = cv2.erode(img_dilation, (5, 1), iterations=1)
    img_dilation = cv2.dilate(img_erode, (7, 7), iterations=5)
    img_blur2 = cv2.GaussianBlur(img_dilation, (31, 31), 5)
    cx, cy = corners_good_features(img_blur2, img2_corners)
    if np.shape(cx)[0] == 4:
        coordinates = np.column_stack((cx, cy))
        coordinates = arrange_points(coordinates)
        cv2.putText(image, "1", (int(coordinates[0][0]), int(coordinates[0][1])), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "2", (int(coordinates[1][0]), int(coordinates[1][1])), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "3", (int(coordinates[2][0]), int(coordinates[2][1])), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "4", (int(coordinates[3][0]), int(coordinates[3][1])), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (255, 255, 0), 1, cv2.LINE_AA)
        return coordinates
    else:
        return None


def arrange_points(coords):
    photo = [[640, 480]]
    zero = [[0, 0]]
    dist0 = np.linalg.norm(coords[0] - zero)
    dist1 = np.linalg.norm(coords[1] - zero)
    dist2 = np.linalg.norm(coords[2] - zero)
    dist3 = np.linalg.norm(coords[3] - zero)
    distances = np.array([dist0, dist1, dist2, dist3])
    first = coords[np.argmin(distances)]
    dist0 = np.linalg.norm(coords[0] - photo)
    dist1 = np.linalg.norm(coords[1] - photo)
    dist2 = np.linalg.norm(coords[2] - photo)
    dist3 = np.linalg.norm(coords[3] - photo)
    distances2 = np.array([dist0, dist1, dist2, dist3])
    third = coords[np.argmin(distances2)]
    dist0 = np.linalg.norm(coords[0] - [[photo[0][0], 0]])
    dist1 = np.linalg.norm(coords[1] - [[photo[0][0], 0]])
    dist2 = np.linalg.norm(coords[2] - [[photo[0][0], 0]])
    dist3 = np.linalg.norm(coords[3] - [[photo[0][0], 0]])
    distances3 = np.array([dist0, dist1, dist2, dist3])
    second = coords[np.argmin(distances3)]
    dist0 = np.linalg.norm(coords[0] - [[0, photo[0][1]]])
    dist1 = np.linalg.norm(coords[1] - [[0, photo[0][1]]])
    dist2 = np.linalg.norm(coords[2] - [[0, photo[0][1]]])
    dist3 = np.linalg.norm(coords[3] - [[0, photo[0][1]]])
    distances4 = np.array([dist0, dist1, dist2, dist3])
    fourth = coords[np.argmin(distances4)]
    coords = np.array([first, second, third, fourth])
    return coords


def create_trackbars():
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 640, 480)
    cv2.createTrackbar('L-H', "Trackbars", 0, 255, nothing)
    cv2.createTrackbar('L-S', "Trackbars", 6, 255, nothing)
    cv2.createTrackbar('L-V', "Trackbars", 146, 255, nothing)
    cv2.createTrackbar('U-H', "Trackbars", 34, 255, nothing)
    cv2.createTrackbar('U-S', "Trackbars", 255, 255, nothing)
    cv2.createTrackbar('U-V', "Trackbars", 255, 255, nothing)





def trackbar(img, color):
    # For red
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "red":
        lowera = np.array([159, 100, 100])
        uppera = np.array([200, 255, 255])
        lowerb = np.array([0, 190, 90])
        upperb = np.array([10, 255, 175])
        lowerc = np.array([0, 160, 70])
        upperc = np.array([255, 255, 110])
        amask = cv2.inRange(img_hsv, lowera, uppera)
        bmask = cv2.inRange(img_hsv, lowerb, upperb)
        cmask = cv2.inRange(img_hsv, lowerc, upperc)
        mask = amask + bmask + cmask
    if color == "red_hand":
        lowera = np.array([0, 63, 63])
        uppera = np.array([8, 255, 255])
        lowerb = np.array([150, 65, 65])
        upperb = np.array([200, 255, 255])
        maska = cv2.inRange(img_hsv, lowera, uppera)
        maskb = cv2.inRange(img_hsv, lowerb, upperb)
        mask = maska + maskb
    if color == "yellow":
        lowera = np.array([17, 90, 100])
        uppera = np.array([50, 255, 255])
        mask = cv2.inRange(img_hsv, lowera, uppera)
    if color == "yellow_hand":
        lowera = np.array([11, 109, 117])
        uppera = np.array([40, 255, 255])
        mask = cv2.inRange(img_hsv, lowera, uppera)
    if color == "green":
        lowera = np.array([40, 40, 20])
        uppera = np.array([80, 170, 170])
        mask = cv2.inRange(img_hsv, lowera, uppera)
    if color == "green_hand":
        lowera = np.array([45, 62, 35])
        uppera = np.array([82, 255, 255])
        mask = cv2.inRange(img_hsv, lowera, uppera)
    if color == "track":
        l_h = cv2.getTrackbarPos('L-H', "Trackbars")
        l_s = cv2.getTrackbarPos('L-S', "Trackbars")
        l_v = cv2.getTrackbarPos('L-V', "Trackbars")
        u_h = cv2.getTrackbarPos('U-H', "Trackbars")
        u_s = cv2.getTrackbarPos('U-S', "Trackbars")
        u_v = cv2.getTrackbarPos('U-V', "Trackbars")
        lowerb = np.array([l_h, l_s, l_v])
        upperb = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(img_hsv, lowerb, upperb)
    return mask


def detect_cubes(img, camMatrix, dist):
    frame = img
    mask_r = trackbar(frame, "red")
    mask_g = trackbar(frame, "green")
    mask_y = trackbar(frame, "yellow")
    cx_r, cy_r = center_point(mask_r, frame)
    cx_y, cy_y = center_point(mask_y, frame)
    cx_g, cy_g = center_point(mask_g, frame)
    coords_r = np.array([cx_r, cy_r])
    coords_y = np.array([cx_y, cy_y])
    coords_g = np.array([cx_g, cy_g])
    coords_r = homography(coords_r, 890, camMatrix, dist)
    coords_y = homography(coords_y, 890, camMatrix, dist)
    coords_g = homography(coords_g, 890, camMatrix, dist)
    if cx_g == 0 and cy_g == 0:
        coords_g = None
    if cx_r == 0 and cy_r == 0:
        coords_r = None
    if cx_y == 0 and cy_y == 0:
        coords_y = None
    return coords_r, coords_y, coords_g


def homography(coords, zc, camMatrix, dist):
    H_fc2o = [[-1, 0, 0, 270],
              [0, 1, 0, 570],
              [0, 0, -1, 860],
              [0, 0, 0, 1]]
    camMatrix = np.linalg.inv(camMatrix)
    coords = np.append(coords, [1], axis=0)
    coords = np.transpose(coords)
    camera_coords = np.dot(camMatrix, coords) * 890
    camera_coords = np.append(camera_coords, [1], axis=0)
    real_coords = np.dot(H_fc2o, camera_coords)
    return np.array([real_coords[0], real_coords[1]])


def center_point(mask, frame_to_draw):
    result = cv2.bitwise_and(frame_to_draw, frame_to_draw, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, peri * 0.02, True)
                cv2.drawContours(frame_to_draw, cnt, -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + (w // 2), y + (h // 2)
                cv2.rectangle(frame_to_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame_to_draw, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame_to_draw, f'{cx}, {cy}', (cx + 20, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (108, 125, 24),
                            1, cv2.LINE_AA)
                return cx, cy
    return 0, 0


def axis_draw(rVec, tVec, camMat, image):
    image_axis = image.copy()
    axis_length = 50

    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])

    axis_image_points, _ = cv2.projectPoints(axis_points, rVec, tVec, camMat, None)
    origin_point = (int(axis_image_points[0][0][0]), int(axis_image_points[0][0][1]))
    cv2.line(image_axis, origin_point, (int(axis_image_points[1][0][0]), int(axis_image_points[1][0][1])), (0, 0, 255),
             2)  # X-axis (red)
    cv2.line(image_axis, origin_point, (int(axis_image_points[2][0][0]), int(axis_image_points[2][0][1])), (0, 255, 0),
             2)  # Y-axis (green)
    cv2.line(image_axis, origin_point, (int(axis_image_points[3][0][0]), int(axis_image_points[3][0][1])), (255, 0, 0),
             2)  # Z-axis (blue)
    # cv2.putText(image_axis, 'hola', (320, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA)
    return image_axis


def pnpSolve(img,corner, camMat, disCo, hg2o, hc2g):
    blank = np.array([0, 0, 0, 1])
    centardito = np.array([25, 25, 0, 1])
    object_points = np.array([[0, 0, 0], [50, 0, 0], [50, 50, 0], [0, 50, 0]], dtype=np.float32)

    success, rotation_vec, translation_vec = cv2.solveP3P(object_points, corner, camMat, disCo,
                                                              flags=cv2.SOLVEPNP_P3P)
    img2 = axis_draw(rotation_vec[0], translation_vec[0], camMat, img)

    x = translation_vec[0][0]
    y = translation_vec[0][1]
    z = translation_vec[0][2]
    rotation_mat, _ = cv2.Rodrigues(rotation_vec[0])
    ht2c = np.append(rotation_mat, np.array([x, y, z]), axis=1)
    ht2c = np.append(ht2c, [blank], axis=0)
    ht2g = np.dot(hc2g, ht2c)
    ht2o = np.dot(hg2o, ht2g)
    c2o = np.dot(ht2o, centardito)
    cv2.imshow('nfa', img2)
    cv2.waitKey(1)
    return ht2o, c2o


def gripMove(cmd):
    activation_request = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x73, 0x30])
    open_gripper = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x72, 0x19])
    close_gripper = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x42, 0x29])
    close_50force = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, 0xFF, 0xFF, 0x7F, 0x43, 0x89])
    close_50p_50v_50f = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, 0x7F, 0x7F, 0x7F, 0x23, 0xA1])
    ser = serial.Serial(port='COM8', baudrate=115200, timeout=1,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS)
    if cmd == 1:
        ser.write(close_50p_50v_50f)
        data_raw = ser.readline()
    elif cmd == 0:
        ser.write(open_gripper)
        data_raw = ser.readline()

def get_angles_rpy(r_r):
    r_x = m.atan2(r_r[2, 1], r_r[2, 2])
    r_y = m.atan2(-r_r[2, 0], np.sqrt(1-(r_r[2, 0]**2)))
    r_z = m.atan2(r_r[1, 0], r_r[0, 0])
    return r_x*180/np.pi, r_y*180/np.pi, r_z*180/np.pi


def grasp(centre, ht2o, rPos, c):
    anglesGonza = get_angles_rpy(ht2o)
    robot_move(centre[0], centre[1], 70, 180, 0, anglesGonza[2], c)
    rPos = receive(c)
    robot_move(centre[0], centre[1], centre[2] - 15, 180, 0, anglesGonza[2], c)
    rPos = receive(c)
    # ------------------------------------------------------------------------------
    # Grasping
    gripMove(1)
    # ------------------------------------------------------------------------------
    robot_move(centre[0], centre[1], 150, 180, 0, 90, c)
    rPos = receive(c)

def stackMove(nstack, c):
    robot_move(350, -30, 150, 180, 0, 90, c)
    rPos = receive(c)
    robot_move(350, -30, 15+(nstack*50), 180, 0, 90, c)
    rPos = receive(c)
    gripMove(0)
    robot_move(350, -30, 150, 180, 0, 90, c)
    rPos = receive(c)
    robot_move(350, 150, 150, 180, 0, 90, c)
    rPos = receive(c)


