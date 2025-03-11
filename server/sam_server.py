from flask import Flask, request, jsonify
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor
from io import BytesIO

app = Flask(__name__)

# Global variables for model and first setup
predictor = None
initialized = False

# Load SAM2 model
sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"


# Route to initialize model and load the first mask and RGB image
@app.route('/initialize_model', methods=['POST'])
def initialize_model():
    global predictor, initialized

    if initialized:
        return jsonify({"error": "Model already initialized."}), 400

    # Receive the initial RGB image and mask
    rgb_img_data = request.files['rgb_img'].read()
    mask_data = request.files['mask'].read()

    # Convert the received RGB image and mask to numpy arrays
    rgb_img = np.frombuffer(rgb_img_data, np.uint8)
    rgb_img = cv2.imdecode(rgb_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    mask = np.frombuffer(mask_data, np.uint8)
    mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0  # Normalize mask to be between 0 and 1

    # Initialize SAM2 model
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    # Set the first image and mask for initialization
    ann_frame_idx = 0
    # Unique id for the object to track
    ann_obj_id = 1
    predictor.load_first_frame(rgb_img)

    # Use the initial mask for inference
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask)

    initialized = True
    return jsonify({"message": "Model initialized successfully."}), 200


# Route to process incoming RGB image frames and return mask
@app.route('/process_rgb', methods=['POST'])
def process_rgb():
    global predictor, initialized

    if not initialized:
        return jsonify({"error": "Model not initialized. Please call /initialize_model first."}), 400

    # Receive the RGB image from the request
    rgb_img_data = request.files['rgb_img'].read()
    rgb_img = np.frombuffer(rgb_img_data, np.uint8)
    rgb_img = cv2.imdecode(rgb_img, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Process the image using SAM2 for tracking and generate the mask
    out_obj_ids, out_mask_logits = predictor.track(rgb_img)

    # Combine the masks from the tracking process
    height, width = rgb_img.shape[:2]
    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        all_mask = cv2.bitwise_or(all_mask, out_mask)

    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)

    # Encode the result mask to send back as a response
    _, buffer = cv2.imencode('.png', all_mask)
    return buffer.tobytes(), 200, {'Content-Type': 'image/png'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
