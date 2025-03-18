from ultralytics import YOLO

class PersonDetector:
    def __init__(self, checkpoints_dir="../../yolo/checkpoints/yolo/yolo11n.pt"):
        # Load YOLO model (adjust the path as needed)
        self.yolomodel = YOLO(checkpoints_dir)

    def predict(self, frame, classes=[0], conf=0.5):
        """Detect persons in the given frame and return predictions."""
        results = self.yolomodel.predict(frame, classes=classes, conf=conf)
        return results

    def get_bbox(self, frame):
        """Returns the largest bounding box corresponding to the closest person."""
        # Run detection on the frame
        results = self.predict(frame)

        largest_box = None
        largest_area = 0

        # Loop through the detected objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the object is a person (class '0' is person)
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    area = (x2 - x1) * (y2 - y1)  # Calculate the area of the bounding box

                    # Update if the area is the largest found
                    if area > largest_area:
                        largest_area = area
                        largest_box = (x1, y1, x2, y2, box.conf[0], int(box.cls[0]))  # (x1, y1, x2, y2, confidence, class)

        return largest_box



