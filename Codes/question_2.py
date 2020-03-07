import cv2
import numpy as np
import matplotlib.pyplot as plot


def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])


# function for warping
def warpImage(pts):

    # store points
    tl = pts[0]
    tr = pts[1]
    br = pts[2]
    bl = pts[3]
    rect = np.array([tl, tr, br, bl], dtype="float32")
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
    return warped
def undistortImage(warped):
    # Define Camera Matrix
    mtx =  np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                     [0.000000e+00, 9.019653e+02, 2.242509e+02],
                     [0, 0, 1]])

    # Define distortion coefficients
    dist = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

    # Getting the new optimal camera matrix
    # img = cv2.imread('image0.jpg')
    h, w = warped.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,
                        (w,h))

    # Checking to make sure the new camera materix was properly generated
    # print(newcameramtx)
    
    # Undistorting
    dst = cv2.undistort(warped, mtx, dist, None, newcameramtx)

if __name__ == '__main__':

    # read image
    image = cv2.imread("img.png")

    # call mouse click function
    points = []
    cv2.namedWindow("image", 1)
    cv2.setMouseCallback("image", mouse_click)

    cv2.imshow("image", image )
    cv2.waitKey(0)
    if 0xFF == ord('q'):
        cv2.destroyAllWindows()

    # get warped image
    warped_img = warpImage(points)
    # print(warped.shape)
    cv2.imshow("warped image", warped_img)
    # cv2.imshow("image", image )
    cv2.waitKey(0)
