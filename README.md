# Lane Detection and Prediction
In this project, we aim to do simple Lane Detection to mimic Lane Departure Warning systems used in Self Driving Cars. We have two video sequences that are taken from a self driving car. The approach is to design an algorithm to detect lanes on the road, as well as estimate the road curvature to predict car turns.

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

<p align="center">
  <img width="400" height="228" src="https://github.com/namangupta98/Lane_Detection/blob/master/Reference%20Images/Q1-input.gif">
  <img width="400" height="228" src="https://github.com/namangupta98/Lane_Detection/blob/master/Reference%20Images/Q1-output.gif">
  <br><b>Figure 1 - Video Brightness Enhancement using Histogram Equalization</b><br>
</p>

     Run the following command to see the improved contrast of the image.
    ```
    python3 Q1_histo.py 
    ```
 - **Lane Detection and Prediction**: Using histogram of lane pixels the lanes are detected and warped to perform post-processing on the lanes to seperate yellow lanes and white lanes in the video sequence using HSV color detection. After lanes are detected, the lanes are refined and unwarped back to the video. For prediction, the difference between lane pixels of previous frame and current frame is used. If the difference is positive the lane is turning right, if negative the lane is turning left, else the road is going straight.
 
 <p align="center">
  <img src="https://github.com/namangupta98/Lane_Detection/blob/master/Reference%20Images/Q2-Data-1.gif">
  <br><b>Figure 2 - Lane Detection and Prediction for Data-1</b><br>
</p>  
    
- Type the following for lane_detection and prediction Data 1
```
python3 question_2_naman.py
```
<p align="center">
  <img src="https://github.com/namangupta98/Lane_Detection/blob/master/Reference%20Images/Q2-Data-2.gif">
  <br><b>Figure 3 - Lane Detection and Prediction for Data-2</b><br>
</p>

- Type the following for lane_detection and prediction for Data 2
```
python3 Question_2_Final_Challenge.py
```

## Results

https://drive.google.com/drive/folders/14he1q5mJtwVrM9toGu5lVNH6lYgcHDK7?usp=sharing
