import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# function to inverse warp
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
    warped = cv2.warpPerspective(den_image, np.linalg.inv(M), (crop.shape[1], crop.shape[0]))
    return warped


if __name__ == '__main__':
    # read image
    cap = cv2.VideoCapture("data_1/out.avi")
    ctr = 0

    while True:

        _, frame = cap.read()

        # undistort image
        undistort_img = undistortImage(frame)

        # denoise image
        denoise_img = cv2.fastNlMeansDenoisingColored(undistort_img, None, 10, 10, 7, 21)
        # cv2.imshow('Denoised Image', denoise_img)

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
        # points = []
        points = np.float32([[468, 54], [685, 54], [820, 154], [155, 154]])

        # for mouse click to get four points
        # if ctr == 220:
        #
        #     cv2.namedWindow("frame", 1)
        #     cv2.setMouseCallback("frame", mouse_click)
        #     cv2.imshow('frame', crop)
        #     cv2.waitKey(0)
        #
        #     if 0xff == ord('q'):
        #         break

        # get warped image
        warped_img = warpImage(crop, points)
        cv2.imshow("warped image", warped_img)

        # using Histogram
        # hist = cv2.calcHist([warped_img], [0], None, [256], [0, 256])

        # plotting histogram
        # plot.plot(hist)
        # plot.show()

        # color separation using HSV
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        higher_white = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_white, higher_white)
        cv2.imshow('mask', mask)

        # # pixel count
        pixel_sum = np.sum(mask, axis=0)

        # plot histogram
        # plt.plot(pixel_sum)
        # plt.xlabel('Image Cols')
        # plt.ylabel('Sum of Pixels')
        # plt.show()

        pts_white = []
        for i in range(len(pixel_sum)):
            if pixel_sum[i]:
                for j in range(mask.shape[0]):
                    pts_white.append([i, j])

        pts_white = np.array(pts_white)
        pts_white = pts_white.reshape((-1, 1, 2))
        warped_img = cv2.polylines(warped_img, [pts_white], False, (255, 0, 0))

        cv2.imshow('lines', warped_img)

        # unwarp the image
        inv_warped_image = invWarpImage(warped_img, points)
        # inv_warped_image = cv2.add(crop, inv_warped_image)
        inv_warped_gray = cv2.cvtColor(inv_warped_image, cv2.COLOR_BGR2GRAY)
        _, inv_warped_thresh = cv2.threshold(inv_warped_gray, 0, 250, cv2.THRESH_BINARY_INV)
        fram_bit = cv2.bitwise_and(crop, crop, mask=inv_warped_thresh)
        # lena_warp = cv2.warpPerspective(lena_img, new_homo, (frame.shape[1], frame.shape[0]))
        new_frame = cv2.add(fram_bit, inv_warped_image)
        cv2.imshow('warped', new_frame)

        # ctr += 1
        # print(ctr)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
