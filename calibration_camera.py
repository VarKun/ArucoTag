import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_DICT_PATH = BASE_DIR / "custom_dict.yml"
CALIBRATION_IMAGE_DIR = BASE_DIR / "image_detected"
CALIBRATION_OUTPUT_PATH = BASE_DIR / "ArucoCheckMultiMatrix.npz"


def load_custom_dictionary(marker_size: int = 4) -> cv2.aruco.Dictionary:
    """Load the persisted custom dictionary from disk."""
    if not CUSTOM_DICT_PATH.exists():
        raise FileNotFoundError(
            f"Custom dictionary not found at {CUSTOM_DICT_PATH}. "
            "Run marker_dictionary_made.py before calibrating."
        )

    cv_file = cv2.FileStorage(str(CUSTOM_DICT_PATH), cv2.FILE_STORAGE_READ)
    custom_bytes_list = cv_file.getNode("custom_dictionary").mat()
    cv_file.release()

    if custom_bytes_list is None:
        raise ValueError(
            "The custom dictionary file does not contain a 'custom_dictionary' node."
        )

    return cv2.aruco.Dictionary(custom_bytes_list, marker_size)


def collect_calibration_points(
    aruco_dict: cv2.aruco.Dictionary,
    aruco_parameters: cv2.aruco.DetectorParameters,
    marker_length: float,
):
    """Collect 3D-2D correspondences from the images folder."""
    object_points = []
    image_points = []
    image_size = None

    if not CALIBRATION_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory '{CALIBRATION_IMAGE_DIR}' does not exist."
        )

    marker_template = np.array(
        [
            [0, 0, 0],
            [marker_length, 0, 0],
            [marker_length, marker_length, 0],
            [0, marker_length, 0],
        ],
        dtype=np.float32,
    )

    for image_path in sorted(CALIBRATION_IMAGE_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not load '{image_path.name}', skipping.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_parameters
        )

        if ids is None:
            print(f"[INFO] No markers detected in '{image_path.name}', skipping.")
            continue

        image_size = gray.shape[::-1]

        for marker_corners in corners:
            object_points.append(marker_template.copy())
            image_points.append(marker_corners.reshape(-1, 2).astype(np.float32))

    if not object_points or not image_points:
        raise RuntimeError("No ArUco markers detected in the calibration images.")

    return object_points, image_points, image_size


if __name__ == '__main__':
    dictionary = load_custom_dictionary()
    parameters = cv2.aruco.DetectorParameters()
    marker_length = 40.0

    object_points, image_points, image_size = collect_calibration_points(
        dictionary, parameters, marker_length
    )

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    np.savez(
        str(CALIBRATION_OUTPUT_PATH),
        camMatrix=mtx,
        distCoef=dist,
        rVector=rvecs,
        tVector=tvecs,
        reprojection_error=ret,
    )

    print(f"Camera calibration is completed and saved to '{CALIBRATION_OUTPUT_PATH.name}'.")
