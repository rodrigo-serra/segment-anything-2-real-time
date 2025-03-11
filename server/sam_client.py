import cv2
import requests
import numpy as np

# Flask server URL
FLASK_SERVER_URL_INITIALIZE = 'http://localhost:5000/initialize_model'
FLASK_SERVER_URL_PROCESS = 'http://localhost:5000/process_rgb'

# Path to the video and mask files
video_path = '../notebooks/videos/gabriel/gabriel.mp4'
mask_path = '../notebooks/masks/gabriel/gabriel.png'

# Function to send video frames and mask to the Flask server
def send_initial_data_to_flask(rgb_frame, mask):
    """Send the initial RGB frame and mask to the Flask server for model initialization."""
    _, img_encoded = cv2.imencode('.jpg', rgb_frame)
    img_bytes = img_encoded.tobytes()

    # Open mask image and encode to bytes
    _, mask_encoded = cv2.imencode('.png', mask)
    mask_bytes = mask_encoded.tobytes()

    # Send the initial RGB frame and mask to Flask server for model initialization
    files = {
        'rgb_img': ('frame.jpg', img_bytes, 'image/jpeg'),
        'mask': ('mask.png', mask_bytes, 'image/png')
    }

    response = requests.post(FLASK_SERVER_URL_INITIALIZE, files=files)

    if response.status_code == 200:
        print("Model initialized successfully.")
    else:
        print(f"Failed to initialize model: {response.status_code}, {response.text}")



def send_rgb_frame_to_flask(rgb_frame):
    """Send RGB frame to the Flask server for processing and receive the mask."""
    _, img_encoded = cv2.imencode('.jpg', rgb_frame)
    img_bytes = img_encoded.tobytes()

    # Send the RGB frame to Flask server for mask prediction
    files = {
        'rgb_img': ('frame.jpg', img_bytes, 'image/jpeg')
    }

    response = requests.post(FLASK_SERVER_URL_PROCESS, files=files)

    if response.status_code == 200:
        # Process the mask received from Flask server
        mask = np.frombuffer(response.content, np.uint8)
        mask = cv2.imdecode(mask, cv2.IMREAD_COLOR)
        print("Received mask from Flask server.")
        return mask
    else:
        print(f"Failed to receive mask from Flask server: {response.status_code}, {response.text}")
        return None

def process_video():
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Convert to 3-channel image for sending
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Process the first frame and send to initialize the model
    ret, frame = cap.read()
    if ret:
        print("Sending initial frame and mask to Flask server for model initialization...")
        send_initial_data_to_flask(frame, mask)

    # Process the remaining frames and send them to the Flask server for mask prediction
    frame_idx = 1
    display_mask = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx}...")
        
        # Send each subsequent frame for mask prediction
        mask = send_rgb_frame_to_flask(frame)

        if mask is not None:
            # You can process the mask here, e.g., display or save it
            if display_mask:
                cv2.imshow("Mask", mask)
                cv2.waitKey(1)
            else:
                frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video()
