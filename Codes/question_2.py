import cv2
import numpy as np
import matplotlib.pyplot as plot
import copy

video = cv2.VideoCapture("data_2/challenge_video.mp4")

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])

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

def invWarpImage(den_image, pts):
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
    warped = cv2.warpPerspective(den_image, np.linalg.inv(M), (maxWidth, maxHeight))
    return warped

def undistortImage(warped):
    ### Define Camera Matrix
    mtx = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                    [0.000000e+00, 9.019653e+02, 2.242509e+02],
                    [0, 0, 1]])

    ### Define distortion coefficients
    dist = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

    h, w = warped.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,
                                                      (w, h))

    ### Undistorting
    dst = cv2.undistort(warped, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def hsv(image):
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 183])
    higher_white = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_white, higher_white)
    return mask



while True:
    _, frame = video.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = np.float32([[550,460],[740, 460],[1280,720],[128, 720]])
    # points = np.float32([[451, 81], [698, 77], [798, 178], [214, 182]]) # test points taken from image

    ### Mouse click call
    # points = []
    # cv2.namedWindow("video", 1)
    # cv2.setMouseCallback("video", mouse_click)

    ### warp function call
    warped_img = warpImage(frame, points)

    ### inverse warp function call
    inv_warped_img = invWarpImage(warped_img, points)

    ### undistort the image
    dst = undistortImage(inv_warped_img)

    ### hsv trial
    processed_image = hsv(dst)


    ### Display the output...
    # cv2.imshow("original_video", gray)
    # cv2.imshow("video", warped_img)
    # cv2.imshow("inverse_video", inv_warped_img)
    # cv2.imshow("undistorted_video", dst)
    cv2.imshow('lane_pixel_candidate', processed_image)



    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()