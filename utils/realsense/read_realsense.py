import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the camera to capture depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the camera
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Display the color image
    cv2.imshow("Color", color_image)
    cv2.imshow("Depth", depth_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the pipeline
pipeline.stop()
cv2.destroyAllWindows()
