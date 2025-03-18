import sys
import os
import time
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.realsense.realsense import RealSenseAligner
from utils.yolo.myyoloclass import PersonDetector
from utils.sam2.mysamclass import MySAM2
from utils.track.centroid_computation import compute_3d_position_from_mask_and_depth
from utils.track.ekf import EKF

import cv2
import numpy as np
import torch


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Initialize RealSenseAligner class
aligner = RealSenseAligner(depth_width=640, depth_height=480, color_width=640, color_height=480, frame_rate=30)
K = aligner.read_camera_params()

# Create YOLO object detector
yolo_detector = PersonDetector()

# Set SAM2
sam_detector = MySAM2()

# Script Options
show_img_result = True
show_centroid = True

# Variables
initial_state = [0, 0, 0, 0, 0, 0]
initialize_ekf = True
get_closest_person = True
cx, cy, x, y, z = None, None, None, None, None

# EFK Parameters
# Define process noise covariance (Q)
process_noise_cov=[[1e-1, 0, 0, 0, 0, 0],
                [0, 1e-1, 0, 0, 0, 0],
                [0, 0, 1e-1, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

# Define measurement noise covariance (R) for 3D measurements
measurement_noise_cov = np.eye(3)/10
# measurement_noise_cov = np.eye(6)/10

# Initial covariance matrix
initial_covariance=[[1e-1, 0, 0, 0, 0, 0],
                [0, 1e-1, 0, 0, 0, 0],
                [0, 0, 1e-1, 0, 0, 0],
                [0, 0, 0, 1e-1, 0, 0],
                [0, 0, 0, 0, 1e-1, 0],
                [0, 0, 0, 0, 0, 1e-1]]

# Time step
dt = 1.0



# Warm-up for 2 seconds
warm_up_time = 2
start_time = time.time()
print(f"Warming Up Camera: {warm_up_time} seconds")
while time.time() - start_time < warm_up_time:
    color_image, depth_image = aligner.process_frames()
    if color_image is None or depth_image is None:
        print("Waiting for valid frames during warm-up...")
        continue


while True:
    # Get color and depth frames
    color_image, depth_image = aligner.process_frames()

    # If frames are valid, process and display
    if color_image is not None and depth_image is not None:
        if get_closest_person:
            # Get the closest person bounding box using YOLO once
            largest_bbox = yolo_detector.get_bbox(color_image)

            if largest_bbox:
                get_closest_person = False
                x1, y1, x2, y2, conf, cls = largest_bbox
                sam_detector.frame_init(frame=color_image, bbox=largest_bbox)
        else:
            mask = sam_detector.predict(frame=color_image)
            # Apply mask to rgb img
            color_image = sam_detector.apply_mask2image(frame=color_image, mask=mask)

            # Compute the 3D position from the mask and depth image
            cx, cy, x, y, z = compute_3d_position_from_mask_and_depth(
                depth_image=depth_image, 
                person_mask=mask, 
                camera_intrinsics=K
            )

            # Log the computed 3D position
            if x is not None and y is not None and z is not None:
                if initialize_ekf:
                    # EKF
                    initial_state = [x, y, z, 0, 0, 0]
                    ekf = EKF(process_noise_cov=process_noise_cov,
                            initial_state=initial_state,
                            initial_covariance=initial_covariance,
                            dt=dt)
                    print("EKF Initialized!")
                    initialize_ekf = False
                else:
                    ekf.predict()
                    ekf.update([x, y, z], dynamic_R=measurement_noise_cov)
                    [x, y, z] = ekf.get_state()
                    # print(f"3D Position - X: {x}, Y: {y}, Z: {z}")
                    # Prepare to send data to client
                    position = {'x': x, 'y': y, 'z': z}
                    response = requests.post('http://localhost:5000/person_position', json=position)
                    if response.status_code != 200:
                        print("Failed to send position to server.")
            else:
                print("Failed to compute valid 3D position.")
        
        
        if show_img_result:
            if cx and cy and show_centroid:
                # Draw a circle at the centroid (x, y) on the RGB img
                centroid_color = (0, 255, 0)
                centroid_radius = 10
                cv2.circle(color_image, (int(cx), int(cy)), centroid_radius, centroid_color, -1)

            # Show images in the OpenCV window
            cv2.imshow('SAM2 MASK"', color_image)
            # Wait for user input to close the window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # ESC or 'q' to quit
                cv2.destroyAllWindows()
                break


# Stop the RealSense pipeline when finished
aligner.stop_pipeline()

