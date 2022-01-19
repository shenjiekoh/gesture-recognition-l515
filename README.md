# Using Intel Realsense L515 to Recognize Elevator Floor Hand Gestures
## Goals
* RGB and depth image streaming
* Read and save the RGB and depth video on real-time
* Save numpy array of video in depth

<p align = "center">
<img src="https://github.com/SHENJIEKOH/intel-realsense/blob/main/image/lidar_camera_gallery_6.jpg" width=50% height=50%>
</p>

##  How to download Intel Realsense SDK 2.0
* Step 1: Go to **Intel Realsense L515** website.

  [Intel Realsense 515](https://www.intelrealsense.com/lidar-camera-l515/)

![](image/IntelRealsense.PNG)
* Step 2: Click **Developers** and find **SDK 2.0**, click it.
* Step 3: Scroll down and download **Intel Realsense Viewer**.
![](image/Viewer.PNG)
* Step 4: Run program, and you have finished downloading.

## Coding environment setting
* Step 1: Download Anaconda (Individual Edition).

  [Download Anaconda](https://www.anaconda.com/products/individual)

* Step 2: Click **Download** and wait for installing.
![](image/Anaconda.PNG)
* Step 3: After installing Anaconda, go to **Anaconda Navigator** and install **Spyder**.
* Step 4: Now, we are going to create an environment that run in Python 3.7.  Open **Anaconda Prompt**.
* Step 5: Enter ```conda env list``` to check you current available environment, then enter ```conda create --name **your name** python=3.7```.
![](image/Prompt.PNG)
* Step 6: Follow the instructions to finish the installation.  When the installation is finished, you will see this environment in **Anaconda Navigator**.  Install **Spyder** again, you have done installing you coding enviroment.
![](image/env.png)

## Code Source
Some of the codes are based on many authors, including:
* streaming_get_numpy.py is based on https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples
* dataset.py is based on https://github.com/piergiaj/pytorch-i3d and https://github.com/aladdinpersson/Machine-Learning-Collection
* pytorch_i3d.py and train_model.py are based on https://github.com/piergiaj/pytorch-i3d 

## Usage of Codes
* streaming_get_numpy.py: Capturing video from Intel Realsense L515 LiDAR Camera, then convert the datas into RGB/Depth mp4 files and npy files
* convert_grayscale.py: Processing the datas and convert them into trainable datas
* dataset.py: Make dataset (including training set and validation set) from csv files and npy files
* pytorch_i3d.py: The model structure of I3D model (based on "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset")
* train_model.py: Train the model

## The datasets
mp4 files include RGB and depth colormap videos
npy files include 8-bit RGB datas and 16-bit depth datas

**The meaning of mp4 and npy filename: modality_subject_floor_trial**



https://user-images.githubusercontent.com/81953068/150064757-0e488cfa-f794-4f2d-b0dc-c04b11ceb1e7.mp4



https://user-images.githubusercontent.com/81953068/150064774-cea91099-9606-4f16-ac9d-ed1040450cdb.mp4

After processing, the data should be in the size of 90 frames*224 pixels*224 pixels
