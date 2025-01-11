import torch
from torchvision.transforms import functional as F

# Define the COCO classes
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ObjectDetector:
    def __init__(self, model):
        self.model = model
        self.object_id_counter = 0
        self.detected_objects = []

    def detect_objects(self, frame):
        # Convert the frame to a tensor
        frame_tensor = F.to_tensor(frame).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            predictions = self.model(frame_tensor)

        # Process predictions
        self.process_predictions(predictions)

    def process_predictions(self, predictions):
        for i in range(len(predictions[0]['boxes'])):
            box = predictions[0]['boxes'][i].numpy()
            score = predictions[0]['scores'][i].item()
            label = predictions[0]['labels'][i].item()

            if score > 0.5:  # Confidence threshold
                self.object_id_counter += 1
                self.detected_objects.append({
                    "object_id": self.object_id_counter,
                    "label": COCO_INSTANCE_CATEGORY_NAMES[label],
                    "score": score,
                    "box": box.tolist()
                })

    def get_detected_objects(self):
        return self.detected_objects
