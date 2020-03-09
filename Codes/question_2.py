import cv2
import numpy as np
import matplotlib.pyplot as plot


def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])


# function for warping
def warpImage(den_image, pts):
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
    warped = cv2.warpPerspective(den_image, M, (maxWidth, maxHeight))
    return warped


# function to undistort image
def undistortImage(warped):
    # Define Camera Matrix
    mtx = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                    [0.000000e+00, 9.019653e+02, 2.242509e+02],
                    [0, 0, 1]])

    # Define distortion coefficients
    dist = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

    # Getting the new optimal camera matrix
    # img = cv2.imread('image0.jpg')
    h, w = warped.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))

    # Undistorting
    dst = cv2.undistort(warped, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    # cv2.imshow('Undistorted Image', dst)
    # cv2.waitKey(0)
    return dst


if __name__ == '__main__':
    # read image
    image = cv2.imread("data_1/data/0000000220.png")

    # undistort image
    undistort_img = undistortImage(image)

    # denoise image
    denoise_img = cv2.fastNlMeansDenoisingColored(undistort_img, None, 10, 10, 7, 21)
    # cv2.imshow('Denoised Image', denoise_img)

    # color seperation using HSV
    # hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0, 0, 255])
    # higher_white = np.array([255, 255, 255])
    # mask = cv2.inRange(hsv, lower_white, higher_white)
    # cv2.imshow('mask', mask)

    # threshold
    # _, thresh = cv2.threshold(warped_img, 250, 255, cv2.THRESH_BINARY)
    # cv2.imshow('lane pixel candidates', thresh)

    # extract edges
    # edges = cv2.Canny(denoise_img, 100, 200)
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)

    # crop real image
    crop = denoise_img[160:372, 0:1281]
    # cv2.imshow('ROI', crop)

    # call mouse click function
    # points = np.float32([[485, 309], [808, 304], [1114, 480], [80, 473]])
    # points = np.float32([[481, 235], [717, 236], [864, 351], [249, 341]])
    points = np.float32([[451, 81], [698, 77], [798, 178], [214, 182]])
    # points = []
    # cv2.namedWindow("image", 1)
    # cv2.setMouseCallback("image", mouse_click)

    # cv2.imshow("image", crop)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # get warped image
    warped_img = warpImage(crop, points)
    cv2.imshow("warped image", warped_img)

    # using Histogram
    hist = cv2.calcHist([warped_img], [0], None, [256], [0, 256])

    # plotting histogram
    plot.plot(hist)
    plot.show()

    cv2.waitKey(0)
