import time
import cv2
import numpy as np
import sys

# Define the available ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Load camera calibration data
calibration_path = "../Apriltage/ArucoCheckMultiMatrix.npz"
calibration_data = np.load(calibration_path)
mtx = calibration_data["camMatrix"]
dist = calibration_data["distCoef"]

# Load custom dictionary
cv_file = cv2.FileStorage("custom_dict.yml", cv2.FILE_STORAGE_READ)
custom_bytes_list = cv_file.getNode("custom_dictionary").mat()
cv_file.release()

marker_size = 4
custom_dictionary = cv2.aruco.Dictionary(custom_bytes_list, marker_size)

desired_aruco_dictionary = custom_dictionary
aruco_parameters = cv2.aruco.DetectorParameters()
aruco_parameters.adaptiveThreshConstant = 7
aruco_parameters.minMarkerPerimeterRate = 0.04
aruco_parameters.maxErroneousBitsInBorderRate = 0.35
aruco_parameters.minCornerDistanceRate = 0.05
aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Capture video stream from the camera
video = cv2.VideoCapture(0)
time.sleep(2.0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FPS, 60)

def get_direction_character(x_coord, case):
    num_chars = 26
    scale = num_chars / 1280
    char_index = int(x_coord * scale)

    if case == 'upper':
        return chr(char_index + ord('A'))

    else:
        return chr(char_index + ord('a'))

def find_gate(corners, ids):
    if len(ids) < 2:
        return None, None

    markers = []
    for i in range(len(ids)):
        markers.append((ids[i], corners[i]))

    gates = []
    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            id1, corner1 = markers[i]
            id2, corner2 = markers[j]
            if (id1 == 0 and id2 == 1) or (id1 == 1 and id2 == 0):
                area1 = cv2.contourArea(corner1)
                area2 = cv2.contourArea(corner2)
                if abs(area1 - area2) / max(area1, area2) < 0.5:
                    center1 = np.mean(corner1.reshape(4, 2), axis=0)
                    center2 = np.mean(corner2.reshape(4, 2), axis=0)
                    if abs(center1[1] - center2[1]) < 150:
                        gates.append((corner1, corner2))
    if gates:
        return gates[0]
    else:
        return None, None

current_case = 'lower'
previous_gate_size = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=desired_aruco_dictionary,
                                                     parameters=aruco_parameters)
    print("[INFO] detected {} markers".format(len(corners)))

    if ids is not None and len(ids) > 0:
        ids = ids.flatten()
        marker1, marker2 = find_gate(corners, ids)

        if marker1 is not None and marker2 is not None:

            corners1 = marker1.reshape(4, 2)
            corners2 = marker2.reshape(4, 2)
            mid_point = (corners1.mean(axis=0) + corners2.mean(axis=0)) / 2
            cX, cY = int(mid_point[0]), int(mid_point[1])

            area1 = cv2.contourArea(marker1)
            area2 = cv2.contourArea(marker2)
            largest_area = max(area1, area2)


            if largest_area < previous_gate_size * 0.5:
                current_case = 'upper' if current_case == 'lower' else 'lower'

            direction_char = get_direction_character(cX, current_case)
            print(f"Direction Character: {direction_char}")

            previous_gate_size = largest_area


            cv2.polylines(frame, [marker1.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)  # 黄色
            cv2.polylines(frame, [marker2.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)  # 黄色
            cv2.circle(frame, (cX, cY), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"Gate:{direction_char}", (cX - 20, cY - 10), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Size:{round(largest_area/1000, 2)}", (cX, cY + 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)


        for i in range(len(corners)):
            corner = corners[i].reshape(4, 2)
            cX = int(corner[:, 0].mean())
            cY = int(corner[:, 1].mean())
            if not (np.array_equal(corners[i], marker1) or np.array_equal(corners[i], marker2)):
                cv2.polylines(frame, [corners[i].astype(np.int32)], True, (0, 0, 255), 4, cv2.LINE_AA)  # 红色
            cv2.putText(frame, f"ID:{ids[i]} X:{cX}", (cX - 20, cY - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        print("No markers detected")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
video.release()
cv2.destroyAllWindows()
