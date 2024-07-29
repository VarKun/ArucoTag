import cv2
import numpy as np

image = cv2.imread("image_detected/right.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

num_markers = 50
marker_size = 4
custom_dictionary = cv2.aruco.extendDictionary(num_markers,marker_size)

aruco_parameters = cv2.aruco.DetectorParameters()

corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray_image, custom_dictionary, parameters=aruco_parameters)

if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    print("Detected IDs:", ids)
else:
    print("No tags detected")

cv_file = cv2.FileStorage("custom_dict.yml", cv2.FILE_STORAGE_WRITE)
cv_file.write("custom_dictionary", custom_dictionary.bytesList)
cv_file.release()

cv2.imshow("Detected Aruco Markers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
