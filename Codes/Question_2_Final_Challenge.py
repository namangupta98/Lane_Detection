import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    warped = cv2.warpPerspective(crop, M, (maxWidth, maxHeight))
    return warped, M, maxWidth, maxHeight


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
    cap = cv2.VideoCapture("data_2/challenge_video.mp4")
    ctr = 0

    while True:

        # read frames
        _, frame = cap.read()

        # using Histogram
        # hist = cv2.calcHist([thresh], [0], None, [256], [0, 256])

        # plotting histogram
        # plot.plot(hist)
        # plot.show()

        # undistort image
        undistort_img = undistortImage(frame)

        # denoise image
        denoise_img = cv2.fastNlMeansDenoisingColored(undistort_img, None, 10, 10, 7, 21)
        # cv2.imshow('Denoised Image', denoise_img)

        # crop real image
        crop = denoise_img[350:720, 0:1280]
        # cv2.imshow('ROI', crop)

        # call mouse click function
        # points = np.float32([[530, 522], [793, 514], [951, 613], [347, 625]])
        # points = np.float32([[463, 115], [860, 119], [1098, 216], [285, 206]])
        points = np.float32([[538, 64], [795, 67], [1197, 158], [247, 154]])
        # points = []

        # for mouse click to get four points
        # if ctr == 5:
        #
        #     cv2.namedWindow("frame", 1)
        #     cv2.setMouseCallback("frame", mouse_click)
        #     cv2.imshow('frame', crop)
        #     cv2.waitKey(0)
        #
        #     if 0xff == ord('q'):
        #         break

        # get warped image
        warped_img, M, maxWidth, maxHeight = warpImage(points)
        # cv2.imshow("warped image", warped_img)
        # cv2.waitKey(0)

        # extract edges
        # edges = cv2.Canny(denoise_img, 100, 200)
        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)

        # threshold
        # gray = cv2.cvtColor(denoise_img, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # cv2.imshow('lane pixel candidates', thresh)
        # cv2.waitKey(0)

        # yellow color separation using HSV
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([0, 52, 127])
        higher_yellow = np.array([26, 156, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, higher_yellow)
        # cv2.imshow('yellow mask', yellow_mask)

        # white color separation using HSV
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 183])
        higher_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hsv, lower_white, higher_white)
        # cv2.imshow('white mask', white_mask)

        # final mask
        mask = cv2.add(yellow_mask, white_mask)
        # cv2.imshow('mask', mask)

        # pixel count
        pixel_sum_yellow = np.sum(yellow_mask, axis=0)
        pixel_sum_white = np.sum(white_mask, axis=0)

        # extracting location of pixels
        pts_yellow = []
        for i in range(len(pixel_sum_yellow)):
            if pixel_sum_yellow[i]:
                for j in range(yellow_mask.shape[0]):
                    pts_yellow.append([i, j])

        pts_yellow = np.array(pts_yellow)

        pts_white = []
        for i in range(len(pixel_sum_white)):
            if pixel_sum_white[i]:
                for j in range(white_mask.shape[0]):
                    pts_white.append([i, j])

        # plot histogram
        # plt.plot(pixel_sum)
        # plt.xlabel('Image Cols')
        # plt.ylabel('Sum of Pixels')
        # plt.show()

        # curve on the image
        curve_coeff = np.polyfit(pts_yellow[:, 0], pts_yellow[:, 1], 2)
        y_yellow = np.poly1d(curve_coeff)
        y_new = [y_yellow(pts_yellow[:, 0])]
        y_new = np.array(y_new)
        y_new = y_new.reshape((y_new.shape[1], 1))

        # line on the image
        pts_yellow = np.array(pts_yellow)
        pts_yellow = pts_yellow.reshape((-1, 1, 2))
        warped_img = cv2.polylines(warped_img, [pts_yellow], False, (0, 0, 255))
        # for i in range(len(pts_yellow)):
        #     pt = pts_yellow[i]
        #     cv2.circle(warped_img, tuple(pt), 1, [0, 0, 255])

        pts_white = np.array(pts_white)
        pts_white = pts_white.reshape((-1, 1, 2))
        warped_img = cv2.polylines(warped_img, [pts_white], False, (255, 0, 0))
        # for i in range(len(pts_white)):
        #     pt = pts_white[i]
        #     cv2.circle(warped_img, tuple(pt), 1, [255, 0, 0])

        cv2.imshow('Homography', warped_img)

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
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
