# Lane Detection and Prediction
This project - a software written in python and opencv library helps in lane detection for autonomous cars.

## Author
- [Naman Gupta](https://github.com/namangupta98/)
- [Pruthvikumar Sanghavi](https://github.com/Pruthvi-Sanghavi/)
- [Amoghavarsha Prasanna](https://github.com/AmoghavarshaP)

## Install Dependencies

- [Python3](https://docs.python-guide.org/starting/install3/linux/)
- [OpenCV-Python](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
- [Numpy](https://docs.scipy.org/doc/numpy/user/install.html)
- [Matplotlib](https://matplotlib.org/users/installing.html)

## Run Instructions
Navigate to your workspace.
```
cd <Workspace>
```
Clone this repository and change the directory to Lane_Detection/Codes folder.
```
git clone https://github.com/namangupta98/Lane_Detection.git
cd Lane_Detection/Codes
```
### Project Sections
The project has three sections.
- **Video Quality Enhancement**: Here, we aim to improve the quality of the video sequence provided of a highway during night. Most of the computer vision pipelines for lane detection or other self-driving tasks require good lighting conditions and color information for detecting good features. A lot of pre-processing is required in such scenarios where lighting conditions are poor. Using the knowledge of [histogram equalization](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html) the video is enhanced.

     Run the following command to see the improved contrast of the image.
    ```
    python3 Q1_histo.py 
    ```
- Type the following for lane_detection and prediction data 1
```
python3 question_2_naman.py
```
- Type the following for lane_detection and prediction for data 2
```
python3 Question_2_Final_Challenge.py
```

## Results

https://drive.google.com/drive/folders/14he1q5mJtwVrM9toGu5lVNH6lYgcHDK7?usp=sharing
