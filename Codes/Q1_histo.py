import cv2
import numpy as np

# def EqualizeHistogram(frame):
#     new_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     new = clahe.apply(new_img)
#     return cv2.cvtColor(new,cv2.COLOR_GRAY2BGR)
#     # clahe= cv2.createCLAHE()
#     # new = clahe.apply(new_img)
#     # new_img = cv2.equalizeHist(new_img)
#     # return cv2.cvtColor(new_img,cv2.COLOR_GRAY2BGR)

# def EqualizeHistogram(frame):
#     new_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     new_img[:,:,2] = clahe.apply(new_img[:,:,2])
#     return cv2.cvtColor(new_img,cv2.COLOR_HSV2BGR)


def EqualizeHistogram(frame):
    new_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(new_img)
    clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    new_img_H = clahe.apply(H)
    new_img_S = clahe.apply(S)
    new_img_V = clahe.apply(V)
    new_img1 = cv2.merge((new_img_H,new_img_S,new_img_V))
    return cv2.cvtColor(new_img1,cv2.COLOR_HSV2BGR)


file='output_histogram4.avi'
writer = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*"MJPG"), 30,(1920, 1080))

cap = cv2.VideoCapture('Night Drive - 2689.mp4')

while(True):
    ret, frame = cap.read()
    if ret == True:
        #addwweight method:
        # gaussian=cv2.GaussianBlur(frame,(7,7),0)
        # frame1= cv2.addWeighted(frame, 3,gaussian,2, 0)
        frame1= EqualizeHistogram(frame)
        frame_1=cv2.resize(frame,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
        frame_new=cv2.resize(frame1,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
        writer.write(frame1)
        both = np.concatenate((frame_1,frame_new), axis=1)
        cv2.imshow('Original and processed:Histogram Equalization', both)
        # cv2.imshow('original',frame_1)
        # cv2.imshow('Histogram Equalization', frame_new)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break

cap.release()
cv2.destroyAllWindows()
