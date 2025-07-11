# pose_utils.py

import numpy as np
import cv2

def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def classify_shot(keypoints):
    try:
        shoulder = keypoints[5]
        elbow = keypoints[7]
        wrist = keypoints[9]
        angle = compute_angle(shoulder, elbow, wrist)
        if angle < 100:
            return "Forehand"
        elif angle > 140:
            return "Backhand"
        else:
            return "Neutral"
    except:
        return "Unknown"

def draw_corner_info_box(frame, player_id, pose, norm_cx, norm_cy, top_left=(20, 20)):
    x, y = top_left
    w, h = 280, 100
    player_name = f"Player {chr(65 + player_id)}"

    # Background and border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)

    # Text
    cv2.putText(frame, player_name, (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Pose: {pose}", (x + 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"X Yolo: {norm_cx:.4f}, Y Yolo: {norm_cy:.4f}", (x + 10, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    return frame
