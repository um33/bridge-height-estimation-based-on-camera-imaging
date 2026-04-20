import cv2
import os

# Step 1: extract frames from your calibration video
video_path = "video(calib).MOV"   # record this specifically for calibration
output_dir = "calib_images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
print(f"Video resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

frame_idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # save every 10th frame — gives variety without too many similar frames
    if frame_idx % 10 == 0:
        path = f"{output_dir}/frame_{frame_idx:04d}.jpg"
        cv2.imwrite(path, frame)
        saved += 1
    frame_idx += 1

cap.release()
print(f"Saved {saved} frames from {frame_idx} total")