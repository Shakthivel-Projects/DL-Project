import cv2
import torch
import torch.nn as nn
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models

# Load the YOLO model (adjust path to your model weights if needed)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load YOLOv5 small model

class FusionResNet(nn.Module):
    def __init__(self, num_classes=80):  # Adjust num_classes based on your dataset
        super(FusionResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # You can choose another ResNet version as needed
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Adjusting for the number of classes

    def forward(self, x):
        return self.resnet(x)

# Instantiate the FusionResNet model
num_classes = 80  # Adjust based on your dataset
fusion_model = FusionResNet(num_classes=num_classes)

# Load the state dictionary into the model
fusion_model.load_state_dict(torch.load(r"C:\Project\fusionresnet_cctv.pth", map_location=torch.device('cpu')))
fusion_model.eval()  # Set the model to evaluation mode

# Initialize the DeepSORT tracker
tracker = DeepSort(max_age=30)

class TrainData:
    def __init__(self, classes):
        self.classes = classes

# Example list of classes based on your model
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
               'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
               'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
               'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
               'toothbrush']

# Initialize the train_data object
train_data = TrainData(classes=class_names)

# Use the laptop's built-in webcam
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Object detection with YOLO
    results = yolo_model(frame)  # Forward the frame to the YOLO model
    detections = []
    
    # Check if results are available
    if results.xyxy[0] is not None and len(results.xyxy[0]) > 0:
        for detection in results.xyxy[0]:  # Only first image in batch
            xmin, ymin, xmax, ymax, conf, cls = detection
            detections.append(([xmin.item(), ymin.item(), xmax.item(), ymax.item()], conf.item(), int(cls.item())))

    # Object tracking with DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Process each tracked object
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Extract object image, resize, and preprocess for classification
        object_img = frame[y1:y2, x1:x2]
        object_img = cv2.resize(object_img, (224, 224))
        object_img = torch.tensor(object_img).permute(2, 0, 1).float().unsqueeze(0).to('cuda')

        # Use FusionResNet to classify the object
        with torch.no_grad():
            outputs = fusion_model(object_img)
            _, predicted_class = torch.max(outputs, 1)

        # Display results on frame
        label = f"ID {track_id}, Class {train_data.classes[predicted_class.item()]}"  # Ensure train_data is defined
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Real-Time Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
