import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Write the video in format .mp4 or .avi
rgb = cv2.VideoWriter('RGB.mp4', fourcc, 30.0, (640, 480))
depth = cv2.VideoWriter('Depth.mp4', fourcc, 30.0, (640, 480))

# Start streaming
pipeline.start(config)

# Preallocation of memory for lists
depth_list = []
color_list = []

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
# =============================================================================
#     This is not working
#     # Convert current numpy arrays to lists
#     curr_depth = depth_image.tolist()
#     curr_color = color_image.tolist()
#     
#     # Append new datas into lists
#     depth_list.append(curr_depth)
#     color_list.append(curr_color)
# =============================================================================

    depth_data = np.copy(depth_image)
    depth_data = np.expand_dims(depth_data, axis = 2)
    depth_list.append(depth_data)
    color_data = np.copy(color_image)
    color_data = np.expand_dims(color_data, axis = 3)
    color_list.append(color_data)
    
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
    # Write images    
    rgb.write(color_image)
    depth.write(depth_colormap) 

    # Show images
    cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB', color_image)
    cv2.imshow('Depth',depth_colormap)
    if cv2.waitKey(1) & 0xFF == 27:
        rgb.release()
        depth.release()
        depth_numpy = np.concatenate(depth_list,axis = 2)
        color_numpy = np.concatenate(color_list,axis = 3)
        np.save('Depth Data', depth_numpy)
        np.save('RGB Data', color_numpy)
        break

# =============================================================================
# Not working
# # Convert lists to numpy arrays and save them as .npy
# depth_np = np.array(depth_list)
# color_np = np.array(color_list)
# =============================================================================

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
