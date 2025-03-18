from sam2.build_sam import build_sam2_camera_predictor
import numpy as np
import torch
import cv2

class MySAM2:
    def __init__(self):
        sam2_checkpoint = "../../checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.out_obj_ids = None
        self.out_mask_logits = None

    def frame_init(self, frame, bbox):
        """Initialize frame for prediction with bounding box."""
        self.predictor.load_first_frame(frame)
        bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
        _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox)


    def predict(self, frame):
        """Make a prediction and return the mask."""
        self.out_obj_ids = None
        self.out_mask_logits = None
        
        self.out_obj_ids, self.out_mask_logits = self.predictor.track(frame)
        all_mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
        
        for i in range(len(self.out_obj_ids)):
            out_mask = (self.out_mask_logits[i] > 0.0).permute(1, 2, 0).byte().cuda()
            all_mask = out_mask.cpu().numpy() * 255
        
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        
        return all_mask
    
    def apply_mask2image(self, frame, mask):
        """Apply the mask to the frame."""
        return cv2.addWeighted(frame, 1, mask, 0.5, 0)




