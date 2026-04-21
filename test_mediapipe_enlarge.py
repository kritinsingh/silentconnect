import cv2
import mediapipe as mp
import numpy as np
import urllib.request

# Download a sample 28x28 image from somewhere or just create a synthetic hand shape
# actually let's use an actual mnist-like image if we can, or just draw a stick figure hand
img = np.zeros((224, 224, 3), dtype=np.uint8)

# Draw palm
cv2.circle(img, (112, 150), 30, (200, 150, 150), -1)
# Draw fingers
cv2.line(img, (112, 120), (112, 50), (200, 150, 150), 15) # Middle
cv2.line(img, (90, 125), (80, 60), (200, 150, 150), 15)  # Index
cv2.line(img, (134, 125), (144, 60), (200, 150, 150), 15) # Ring
cv2.line(img, (150, 140), (170, 80), (200, 150, 150), 15) # Pinky
cv2.line(img, (80, 160), (40, 120), (200, 150, 150), 18) # Thumb

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=1
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

res = landmarker.detect(mp_image)
print("Enlarged/Drawn Detected:", len(res.hand_landmarks) if hasattr(res, 'hand_landmarks') else 0)
