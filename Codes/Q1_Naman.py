import cv2
import numpy as np

cap = cv2.VideoCapture('Night_Drive.mp4')

alpha = float(input('Enter Alpha (0.0-3.0): '))
beta = int(input('Enter Beta (0-100): '))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080))

while cap:
    _, frame = cap.read()
    # resize_cap = cv2.resize(frame, (500, 500))
    new_image = np.zeros(frame.shape, frame.dtype)

    new_image = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    cv2.imshow('Original Video', frame)
    cv2.imshow('Enhanced Video', new_image)
    out.write(new_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
