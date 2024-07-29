import cv2


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)


for i in range(20):
    tag = cv2.aruco.generateImageMarker(aruco_dict, i, 400)
    cv2.imshow(f"Tag {i}", tag)
    cv2.imwrite(f"tag_{i}.png", tag)
    cv2.waitKey(0)

cv2.destroyAllWindows()
