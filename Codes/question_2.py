import cv2
import numpy as np
import matplotlib.pyplot as plot

# def mouse_click(event, x, y, flag, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,y)
#         plot.scatter(x,y)
image = cv2.imread("img.png")
# lap1 = cv2.Laplacian(image, cv2.CV_64F)
# cv2.namedWindow("lap1",1)
# cv2.setMouseCallback("lap1", mouse_click)

## check
# image = cv2.line(image, (467,358), (770,358), (255, 0, 0), 2)
# image = cv2.line(image, (770,358), (865,496), (255, 0, 0), 2)
# image = cv2.line(image, (865,496), (260,490), (255, 0, 0), 2)
# image = cv2.line(image, (467,358), (260,490), (255, 0, 0), 2)
rect = np.array([[467,358], [770,358], [865,496], [260,490]], dtype="float32")
tl = [467,358]
tr = [770,358]
br = [865,496]
bl = [260,490]
# rect = (tl,tr,br,bl)
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")
# dst = np.array([[0,0],[199,0],[199,199],[0,199]], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
# print(maxWidth, maxHeight)
warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
print(warped.shape)
cv2.imshow("lap1", warped)
cv2.waitKey(0)
