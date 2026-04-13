import cv2
cap = cv2.VideoCapture("video.MOV")
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))