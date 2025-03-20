import pyrealsense2 as rs
import numpy as np
import json, os

class RealSenseAligner:
    def __init__(self, depth_width=640, depth_height=480, color_width=640, color_height=480, frame_rate=30, clipping_distance_in_meters=1):
        # Cam Intrinsics file path
        # cam_params_rgb.json is meant for rgb and aligned_depth_to_color
        self.cam_params_path = os.path.join(os.path.dirname(__file__), 'cam_params_rgb.json') 

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device and check for RGB Camera
        self.device_product_line = None
        self.found_rgb = False
        self.setup_device()

        # Configure streams
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, frame_rate)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get the depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"Depth Scale is: {self.depth_scale}")

        # Setup clipping distance
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale

        # Create align object for depth to color alignment
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def setup_device(self):
        """Checks if the device has an RGB camera and retrieves its information."""
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break

        if not self.found_rgb:
            print("The demo requires Depth camera with Color sensor")
            raise RuntimeError("RGB Camera not found")

    def process_frames(self):
        """Main streaming loop for processing frames."""
        # Wait for the next frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image


    def stop_pipeline(self):
        self.pipeline.stop()

    
    def read_camera_params(self):
        """Reads camera parameters from a json file."""
        try:
            with open(self.cam_params_path, 'r') as json_file:
                data = json.load(json_file)
            
            k = np.array(data['K'])
            K = [
                    [k[0], k[1], k[2]],
                    [k[3], k[4], k[5]],
                    [k[6], k[7], k[8]]
                ]
            return K
        
        except FileNotFoundError:
            print(f"Error: {self.cam_params_path} not found.")
            raise
