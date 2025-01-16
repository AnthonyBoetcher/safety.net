import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 models
model_custom = YOLO(r'Safety.net\ML_model.pt')  # Raw string
model_pretrained = YOLO(r'Safety.net\yolov8n.pt')  # Raw string

cv2.setNumThreads(1)

# Initialize the live camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))  # Resize to match the model's input size

    # Perform inference using the custom model
    results = model_custom(resized_frame)

    # Extract detections
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract coordinates, confidence, and class labels
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Filter by confidence score (e.g., greater than 0.5)
            if confidence > 0.5:
                # Draw bounding boxes and label on the frame
                label = f"{model_custom.names[class_id]}: {confidence:.2f}"
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Construction Safety Detector', resized_frame)

    # Press 'q' to quit the live feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()