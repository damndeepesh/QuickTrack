import torch
import cv2
import numpy as np
from collections import defaultdict
import streamlit as st

# Add this color_map dictionary before the draw_boxes function
# Extended color map for different classes
color_map = {
    # People and animals
    'person': (0, 0, 255),      # Red
    'dog': (0, 255, 255),       # Cyan
    'cat': (255, 0, 255),       # Magenta
    'bird': (165, 42, 42),      # Brown
    'horse': (128, 0, 0),       # Maroon
    'sheep': (230, 216, 173),   # Beige
    'cow': (112, 128, 144),     # Slate
    
    # Vehicles
    'car': (255, 0, 0),         # Blue
    'truck': (255, 165, 0),     # Orange
    'bicycle': (128, 0, 128),   # Purple
    'motorcycle': (255, 192, 203), # Pink
    'bus': (255, 255, 0),       # Yellow
    'train': (0, 128, 0),       # Dark Green
    'airplane': (70, 130, 180),  # Steel Blue
    'boat': (0, 165, 255),      # Orange-Red
    
    # Objects
    'traffic light': (0, 255, 127),  # Spring Green
    'fire hydrant': (255, 69, 0),    # Red-Orange
    'stop sign': (220, 20, 60),      # Crimson
    'bench': (107, 142, 35),         # Olive
    'chair': (0, 128, 128),          # Teal
    'dining table': (255, 215, 0),   # Gold
    'cell phone': (138, 43, 226),    # Blue Violet
    'laptop': (0, 191, 255),         # Deep Sky Blue
    'keyboard': (255, 127, 80),      # Coral
    'book': (218, 112, 214),         # Orchid
    'clock': (240, 230, 140),        # Khaki
    
    # Sports
    'sports ball': (0, 250, 154),    # Medium Spring Green
    'kite': (255, 240, 245),         # Lavender
    'baseball bat': (188, 143, 143), # Rosy Brown
    'baseball glove': (46, 139, 87), # Sea Green
    
    # Food
    'bottle': (0, 206, 209),         # Turquoise
    'wine glass': (255, 248, 220),   # Cornsilk
    'cup': (147, 112, 219),          # Medium Purple
    'fork': (218, 165, 32),          # Goldenrod
    'sandwich': (210, 105, 30),      # Chocolate
    'pizza': (188, 143, 143),        # Rosy Brown
    
    # Additional objects
    'backpack': (0, 100, 0),         # Dark Green
    'umbrella': (255, 182, 193),     # Light Pink
    'handbag': (219, 112, 147),      # Pale Violet Red
    'tie': (106, 90, 205),           # Slate Blue
    'suitcase': (72, 61, 139),       # Dark Slate Blue
    'frisbee': (32, 178, 170),       # Light Sea Green
    'skis': (135, 206, 250),         # Light Sky Blue
    'snowboard': (176, 224, 230),    # Powder Blue
    'tennis racket': (218, 112, 214),# Orchid
    'surfboard': (0, 139, 139),      # Dark Cyan
    'remote': (143, 188, 143),       # Dark Sea Green
    'mouse': (216, 191, 216),        # Thistle
    'toaster': (255, 222, 173),      # Navajo White
    'sink': (112, 128, 144),         # Slate Gray
    'refrigerator': (47, 79, 79),    # Dark Slate Gray
    'tv': (25, 25, 112),             # Midnight Blue
    'microwave': (0, 139, 139),      # Dark Cyan
    'oven': (160, 82, 45),           # Sienna
    'toothbrush': (199, 21, 133),    # Medium Violet Red
    'scissors': (176, 196, 222),     # Light Steel Blue
}

def load_model(model_path='yolov8x.pt'):
    """Load YOLOv8 model"""
    try:
        from ultralytics import YOLO
        import os
        os.environ['NNPACK_FAST_MATH'] = 'OFF'
        
        # Load the selected model
        model = YOLO(model_path)
        
        # Warmup the model
        model.predict(np.zeros((640, 640, 3)), verbose=False)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def detect_objects(model, frame, conf_threshold=0.5):
    """
    Detect objects in a frame using YOLO with optimized processing
    """
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform detection with optimized settings
    results = model.predict(
        frame_rgb,
        conf=conf_threshold,
        verbose=False,
        device='0' if torch.cuda.is_available() else 'cpu',
        imgsz=1280,  # Increased size for better detection
        iou=0.4,    # Adjusted IOU threshold
        max_det=300,  # Increase maximum detections
        agnostic_nms=True,  # Better handling of objects of different sizes
    )
    
    result = results[0]
    detections = []
    
    if hasattr(result, 'boxes'):
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': confidence
                }
                detections.append(detection)
            except Exception as e:
                continue
    
    return detections

class ObjectTracker:
    def __init__(self):
        self.next_id = 1
        self.object_ids = {}
        self.id_timeout = 30
        self.last_positions = {}
    
    def get_object_id(self, bbox, class_name):
        """Assign or retrieve ID for detected object based on position and IoU"""
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Calculate box area
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        best_iou = 0
        best_id = None
        
        # Check existing objects
        for obj_id, (old_bbox, old_class, last_seen) in list(self.last_positions.items()):
            if last_seen > self.id_timeout:
                del self.last_positions[obj_id]
                continue
            
            # Calculate IoU
            x1 = max(bbox[0], old_bbox[0])
            y1 = max(bbox[1], old_bbox[1])
            x2 = min(bbox[2], old_bbox[2])
            y2 = min(bbox[3], old_bbox[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                old_area = (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])
                union = box_area + old_area - intersection
                iou = intersection / union
                
                if iou > best_iou and iou > 0.3 and class_name == old_class:
                    best_iou = iou
                    best_id = obj_id
        
        if best_id is not None:
            self.last_positions[best_id] = (bbox, class_name, 0)
            return best_id
        
        # If no match found, assign new ID
        new_id = self.next_id
        self.next_id += 1
        self.last_positions[new_id] = (bbox, class_name, 0)
        return new_id
    
    def update_timeouts(self):
        """Update timeout counters for all tracked objects"""
        for obj_id in self.last_positions:
            bbox, class_name, timeout = self.last_positions[obj_id]
            self.last_positions[obj_id] = (bbox, class_name, timeout + 1)

def draw_boxes(frame, detections, tracker):
    """
    Optimized version of drawing bounding boxes and labels with improved visibility
    """
    annotated_frame = frame.copy()
    tracker.update_timeouts()
    
    # Thicker lines and larger text for better visibility
    box_thickness = 3
    text_scale = 0.7
    text_thickness = 2
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        obj_id = tracker.get_object_id(det['bbox'], det['class'])
        
        # Get color with default
        color = color_map.get(det['class'].lower(), (0, 255, 0))
        
        # Create label with better formatting
        label = f"#{obj_id} {det['class']} {det['confidence']:.2f}"
        
        # Draw box with thicker lines
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)
        
        # Improve text background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        
        # Make background rectangle slightly larger
        padding = 5
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - text_height - baseline - padding * 2),
                     (x1 + text_width + padding * 2, y1),
                     color, -1)
        
        # Add white border around the text for better visibility
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            cv2.putText(annotated_frame, label,
                       (x1 + padding + dx, y1 - padding + dy),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                       (0, 0, 0), text_thickness + 1)
        
        # Draw main text
        cv2.putText(annotated_frame, label,
                    (x1 + padding, y1 - padding),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                    (255, 255, 255), text_thickness)
        
        det['id'] = obj_id
    
    return annotated_frame