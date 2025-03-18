from realsense import RealSenseAligner
import cv2
import numpy as np

# Initialize RealSenseAligner class
aligner = RealSenseAligner(depth_width=640, depth_height=480, color_width=640, color_height=480, frame_rate=30)

# Create window for displaying results
cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)

try:
    while True:
        # Get color and depth frames
        color_image, depth_image = aligner.process_frames()

        # If frames are valid, process and display
        if color_image is not None and depth_image is not None:
            # Apply color map to depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Combine color and depth images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images in the OpenCV window
            cv2.imshow('Align Example', images)

            # Wait for user input to close the window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # ESC or 'q' to quit
                cv2.destroyAllWindows()
                break

finally:
    # Stop the RealSense pipeline when finished
    aligner.stop_pipeline()
