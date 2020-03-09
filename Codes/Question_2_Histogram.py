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
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    return warped


# function to undistort image
def undistortImage(warped):
    # Define Camera Matrix
    mtx = np.array([[1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
                     [0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
                     [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

    # Define distortion coefficients
    dist = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])

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
    cap = cv2.VideoCapture("data_2/challenge_video.mp4")
    ctr = 0

    while True:

        # read frames
        _, frame = cap.read()

        # call mouse click function
        points = np.float32([[530, 522], [793, 514], [951, 613], [347, 625]])
        # points = []

        # for mouse click to get four points
        if ctr == 400:

            # cv2.namedWindow("frame", 1)
            # cv2.setMouseCallback("frame", mouse_click)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

            # if 0xff == ord('q'):
            #     break

            # get warped image
            warped_img = warpImage(points)
            cv2.imshow("warped image", warped_img)
            cv2.waitKey(0)

        # using Histogram
        # hist = cv2.calcHist([thresh], [0], None, [256], [0, 256])

        # plotting histogram
        # plot.plot(hist)
        # plot.show()

        # undistort image
        # undistort_img = undistortImage(warped_img)

        # denoise image
        # denoise_img = cv2.fastNlMeansDenoisingColored(undistort_img, None, 10, 10, 7, 21)
        # cv2.imshow('Denoised Image', denoise_img)

        # extract edges
        # edges = cv2.Canny(denoise_img, 100, 200)
        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)

        # crop real image
        # crop = image[190:512, 0:1392]
        # cv2.imshow('ROI', crop)

        # threshold
        # gray = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # cv2.imshow('lane pixel candidates', thresh)
        # cv2.waitKey(0)

        # color seperation using HSV
        # hsv = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2HSV)
        # lower_white = np.array([0, 0, 255])
        # higher_white = np.array([255, 255, 255])
        # mask = cv2.inRange(hsv, lower_white, higher_white)
        # cv2.imshow('mask', mask)

        ctr += 1
        print(ctr)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
