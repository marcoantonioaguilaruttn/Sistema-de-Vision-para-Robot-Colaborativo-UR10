import depthai as dai
import numpy as np
import cv2 as cv

def intrinsicosCamara():
    with dai.Device() as device:
        calibData = device.readCalibration()
    
        intrinsics = calibData.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            1280,
            720
        )

        camera_matrix = np.array(intrinsics)
    
        dist_coeffs = np.array(
            calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        )
    
        print("Camera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs)

    return camera_matrix, dist_coeffs