import cv2
import numpy as np
video = cv2.VideoCapture("Night Drive - 2689.mp4")
gamma = 0.3
while True:
    _, frame = video.read()
    new_video = np.empty((1, 256), np.uint8)
    for i in range(256):
        new_video[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(frame, new_video)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Figure", res)
    cv2.imshow("Figure1", frame)
    output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (res.shape[1], res.shape[0]))

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
video.release()
cv2.destroyAllWindows()