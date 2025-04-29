# Video Test (for local system)
import os
import time
import numpy as np
import cv2
import tensorflow as tf

# Constants
IMAGE_SIZE = 224
FRAMES_PER_VIDEO = 10

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "IP")
MODEL_PATH = os.path.join(desktop_path, "deepfake_detector_final.keras")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Unable to open video file!")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, max(1, total_frames - 1), num_frames, dtype=int)
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return np.array(frames)

def predict_video(video_path):
    if not os.path.exists(video_path):
        print("[ERROR] Video file not found!")
        return None, None
    print(f"\n[INFO] Processing video: {video_path}")
    frames = extract_frames(video_path)
    if frames is None or len(frames) == 0:
        print("[ERROR] No frames extracted!")
        return None, None
    frames = np.expand_dims(frames, axis=0)
    start_time = time.time()
    prediction = model.predict(frames)[0][0]
    end_time = time.time()
    label = "FAKE" if prediction >= 0.47 else "REAL"
    confidence = max(prediction, 1 - prediction)
    print(f"\n[RESULT] Video classified as: *{label}*")
    print(f"[CONFIDENCE] {confidence:.2%}")
    print(f"[TIME TAKEN] {end_time - start_time:.2f} sec")
    return label, confidence

if __name__ == "__main__":
    video_path = input("Enter path of the video to check: ").strip()
    label, confidence = predict_video(video_path)
    if label:
        print(f"Classification: {label} ({confidence:.2%} confidence)")
