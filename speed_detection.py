import time
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from nets import nn
from utils import util


# Load YOLO model and class list
model = YOLO('weights/yolov8n.pt')
tracker = nn.DeepSort()

# Line positions and counters
red_line_y = 198
blue_line_y = 268
offset = 6
down = {}
up = {}
counter_down = []
counter_up = []


# Video capture
cap = cv2.VideoCapture('data/highway_mini.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)

    # Prepare input for DeepSort
    detections = results[0].boxes.data.cpu().numpy()  # YOLO detections to numpy
    bbox_xywh = []
    confidences = []
    oids = []

    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        w, h = x2 - x1, y2 - y1
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, w, h])
        confidences.append(conf)
        oids.append(class_id)  # Object class ID

    # Update tracker
    outputs = tracker.update(np.array(bbox_xywh), confidences, oids, frame)

    for output in outputs:
        x1, y1, x2, y2, track_id, track_oid = output
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))  # Format bbox for fancy_bounding_box

        # Downward movement (red to blue)
        if red_line_y - offset < y1 < red_line_y + offset:
            down[track_id] = time.time()

        if track_id in down and blue_line_y - offset < y2 < blue_line_y + offset:
            elapsed_time = time.time() - down[track_id]
            if track_id not in counter_down:
                counter_down.append(track_id)
                distance = 10  # Distance between lines in meters
                speed_ms = distance / elapsed_time
                speed_kh = speed_ms * 3.6

                # Draw bounding box with `fancy_bounding_box`
                frame = util.fancy_bounding_box(frame, bbox, index=track_id)
                cv2.putText(frame, f'Speed: {int(speed_kh)} km/h', (bbox[0], bbox[1] + bbox[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Upward movement (blue to red)
        if blue_line_y - offset < y2 < blue_line_y + offset:
            up[track_id] = time.time()

        if track_id in up and red_line_y - offset < y1 < red_line_y + offset:
            elapsed_time = time.time() - up[track_id]
            if track_id not in counter_up:
                counter_up.append(track_id)
                distance = 10  # Distance between lines in meters
                speed_ms = distance / elapsed_time
                speed_kh = speed_ms * 3.6

                # Draw bounding box with `fancy_bounding_box`
                frame = util.fancy_bounding_box(frame, bbox, index=track_id)
                cv2.putText(frame, f'Speed: {int(speed_kh)} km/h', (bbox[0], bbox[1] + bbox[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display counters and lines
    cv2.putText(frame, f'Going Down: {len(counter_down)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f'Going Up: {len(counter_up)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.line(frame, (0, red_line_y), (frame.shape[1], red_line_y), (0, 0, 255), 2)
    cv2.line(frame, (0, blue_line_y), (frame.shape[1], blue_line_y), (255, 0, 0), 2)

    # Save and display frame
    out.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
