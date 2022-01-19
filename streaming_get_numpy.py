"""
Streaming with IntelRealsense series camera
Getting the RGB/D mp4 video files and their npy files
When running code, then start streaming, finally stop after (max_frame) frames

@author: Shen Jie Koh
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
fourcc = cv2.VideoWriter_fourcc(*'XVID')
floor = 'B2'
trial = 10
subject = 5

video_info = str(subject) + '_' + floor + '_' + str(trial)
rgb = cv2.VideoWriter('../data/video/rgb/rgb_' + video_info + '.mp4', fourcc, 30.0, (640, 360))
depth = cv2.VideoWriter('../data/video/depth/depth_' + video_info + '.mp4', fourcc, 30.0, (640, 360))

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
dist = 250 # in mm
if depth_sensor.supports(rs.option.min_distance):
    depth_sensor.set_option(rs.option.min_distance, dist)
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

depth_list = []
color_list = []
max_frame = 90  # stop after 90 frames
frame = 1
while frame <= max_frame:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Get frame data
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # Convert depth image to colormap
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape
    
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
    
    # Append datas to lists
    depth_data = np.copy(depth_image)
    depth_data = np.expand_dims(depth_data, axis=0)
    depth_list.append(depth_data)
    if depth_colormap_dim != color_colormap_dim:
        color_data = np.copy(resized_color_image)
    else:
        color_data = np.copy(color_image)
    color_data = np.expand_dims(color_data, axis=0)
    color_list.append(color_data)
    
    # Write data to VideoWriter
    if depth_colormap_dim != color_colormap_dim:
        rgb.write(resized_color_image)
        img = np.concatenate((resized_color_image, depth_colormap), axis=1)
    else:
        rgb.write(color_image)
        img = np.concatenate((color_image, depth_colormap), axis=1)
    depth.write(depth_colormap)
    
    # Show image
    cv2.namedWindow('RGB+Depth', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB+Depth', img)
    frame += 1
    
    # Press esc or 'q' to close the image window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
rgb.release()
depth.release()
# Concatenate datas and save them as npy files
depth_numpy = np.concatenate(depth_list, axis = 0)
color_numpy = np.concatenate(color_list, axis = 0)
np.save('../data/raw/depth_16bit/depth_' + video_info, depth_numpy)
np.save('../data/raw/rgb_8bit/rgb_' + video_info, color_numpy)

profile = pipeline.stop()
cv2.destroyAllWindows()