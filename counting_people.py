from ultralytics import YOLO
from nets import nn
import cv2
import torch
from collections import deque
from utils import util

# Initialize deque for tracking objects
data_deque = {}
decision_boundary = 50  # in pixels
data_deque = {}
up_count = 0
down_count = 0
line_y = 300  # Position of the line
deepsort = nn.DeepSort()
model = YOLO("weights/yolov8n.pt")
# Initialize deque for tracking objects


def draw_boxes(image, boxes, object_id, identities):
    global up_count, down_count
    h, w, _ = image.shape

    cv2.line(image, (10, line_y), (w - 10, line_y), (255, 0, 0), 2)

    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(boxes):
        if object_id[i] != 0:
            continue
        x1, y1, x2, y2 = list(map(int, box))
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        index = int(identities[i]) if identities is not None else 0

        if index not in data_deque:
            data_deque[index] = deque(maxlen=2)
        data_deque[index].appendleft(center)

        # Check movement across the line
        if len(data_deque[index]) >= 2:
            prev_y = data_deque[index][1][1]
            curr_y = data_deque[index][0][1]

            if prev_y < line_y <= curr_y:  # Moving down
                down_count += 1
            elif prev_y > line_y >= curr_y:  # Moving up
                up_count += 1

        # Draw fancy bounding box
        bbox = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
        util.fancy_bounding_box(image, bbox, index=index)

        cv2.circle(image, center, 5, (0, 255, 0), -1)

    cv2.putText(image, f'Up: {up_count}', (10, h - 60), 0, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Down: {down_count}', (10, h - 30), 0, 1, (255, 0, 0), 2)


def predict():
    """
    Run prediction and tracking.
    """
    cap = cv2.VideoCapture("test.mp4")  # Open video source

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLOv8 model for inference
        results = model.predict(source=frame, device='cuda')

        for result in results:
            outputs = deepsort.update(
                torch.Tensor(result.boxes.xywh).to('cuda'),
                torch.Tensor(result.boxes.conf).to('cuda'),
                result.boxes.cls, frame
            )
            if len(outputs) > 0:
                draw_boxes(frame, outputs[:, :4], outputs[:, -1], outputs[:, -2])

        # Show the video with tracking
        cv2.imshow("YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict()