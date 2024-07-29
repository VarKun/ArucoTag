import cv2
import numpy as np
import os

cv_file = cv2.FileStorage("custom_dict.yml", cv2.FILE_STORAGE_READ)
custom_bytes_list = cv_file.getNode("custom_dictionary").mat()
cv_file.release()

marker_size = 4
custom_dictionary = cv2.aruco.Dictionary(custom_bytes_list, marker_size)

aruco_Dict = custom_dictionary
aruco_Parameter = cv2.aruco.DetectorParameters()

marker_Length = 40

object_points_3D = []
image_2D = []


imagePath = "image_detected"
files = os.listdir(imagePath)



def checkArucoMarker(files, imagePath, aurcoParamether, arucoDict, markerlength ):
    for file in files:
        image_path = os.path.join(imagePath, file)
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(gray,arucoDict, parameters= aurcoParamether)
        createMarkerShape(corners,ids,rejected,markerlength,gray)
    return gray



def createMarkerShape(corners, ids, rejected, marklength,image_gray):
    if len(corners) > 0 :
        for i in range(len(ids)):
            object_point = [np.array([[0,0,0],[marklength,0,0],[marklength,marklength,0],[0,marklength,0]], dtype= np.float32)]
            object_points_3D.append(object_point)
            image_2D.append(corners[i])


if __name__ == '__main__':

    gray = checkArucoMarker(files=files,imagePath=imagePath,aurcoParamether=aruco_Parameter,arucoDict=aruco_Dict,markerlength=marker_Length)

    object_3D = np.array(object_points_3D).reshape(-1, 3)
    object_2D = np.array(image_2D).reshape(-1, 2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_3D], [object_2D], gray.shape[::-1], None, None)

    calibration_data_path = "../Apriltage"
    np.savez(f"{calibration_data_path}/ArucoCheckMultiMatrix", camMatrix=mtx, distCoef=dist, rVector=rvecs, tVector=tvecs)

    print("Camera calibration is completed and saved.")