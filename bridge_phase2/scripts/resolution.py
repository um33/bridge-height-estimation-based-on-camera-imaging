import cv2

cap = cv2.VideoCapture("video.MOV")  # open the video file for reading
ret, frame = cap.read()
if ret:
    print("frame.shape:", frame.shape)  # prints (height, width, channels)
    print("height:", frame.shape[0])
    print("width:", frame.shape[1])
else:
    print("Could not read frame")
cap.release()