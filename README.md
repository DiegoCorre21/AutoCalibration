# AutoCalibration
To use this code you need to save 6 points in the robot teaching pendant in P1, P2, ..., P6 slots.

In these points the robot hand-camera needs to be looking directly at a ChAruco board to detect it and use it for the calibration.

The ChAruco board we used is a 6x9 in dimensions, with squares = 29.7 mm, markers = 23.0 mm, and uses a 6x6_1000 dictionary, you can change it in the code. You also need to input the name of your camera matrix npz file.

We use an intel realsense D435 camera in RGB mode in the calibration, and use a file called realsense_depth.py for ease of initialization of the camera, if you use other camera, you will need to change it in the code.
We use an Excel file to read the points in the robot, you need to change these points to the x,y,z,Rx,Ry,Rz you saved in P1, P2, ..., P6 slots.
The code in the robot is called Auto_Hand_Calibration.
Once the calibration is done, a npz file called H_cam2grip will be generated.

![si](https://user-images.githubusercontent.com/104682170/235343177-b28640e4-7d35-47ea-9e25-89b46c8685e0.png)

