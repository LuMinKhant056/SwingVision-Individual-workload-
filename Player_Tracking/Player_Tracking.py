# main_pose_tracking.py
# Before run, you need to install ultralytics libraries by uisng following Query 
# You need to create .venv file to ran the project
# install - python -m venv .venv (create venv)
# install -.venv\Scripts\activate
# If you remember the libraries you used, install them like this: pip install numpy opencv-python torch ultralytics
# Optional extras : pip install matplotlib pandas scikit-learn
# Once done, save them to a file: pip freeze > requirements.txt
# Now you can run this script using: python main_pose_tracking.py

from ultralytics import YOLO
import cv2
import os
import numpy as np
from pose_utils import classify_shot, draw_corner_info_box

# Output directory
os.makedirs("VideoOutput", exist_ok=True)

# Load model and video
model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture("VideoInput/video_input3.mp4")

if not cap.isOpened():
    print("ailed to open input video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (width, height)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("VideoOutput/video_output_pose_filtered.mp4", fourcc, fps, output_size)

# Display window
cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimation", 800, 600)

# Define court filtering area
court_top = int(height * 0.2)
court_bottom = int(height * 0.95)
court_left = int(width * 0.05)
court_right = int(width * 0.95)

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)

    for r in results:
        keypoints = r.keypoints.xy
        bboxes = r.boxes.xyxy if r.boxes is not None else []

        for i, person in enumerate(keypoints):
            if i >= len(bboxes):
                continue

            x1, y1, x2, y2 = bboxes[i].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box_height = y2 - y1

            if court_left < cx < court_right and court_top < cy < court_bottom and box_height > 80:
                keypoint_array = person.cpu().numpy()
                label = classify_shot(keypoint_array)

                # Draw keypoints
                for x, y in keypoint_array:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # Draw smart corner box (top-left for Player A only)
                if i == 0:
                    norm_cx = cx / width
                    norm_cy = cy / height
                    frame = draw_corner_info_box(frame, i, label, norm_cx, norm_cy)

    frame = cv2.resize(frame, output_size)
    cv2.imshow("Pose Estimation", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalize
cap.release()
out.release()
cv2.destroyAllWindows()
print("Final video saved to: VideoOutput/video_output_pose_filtered.mp4")
