import cv2
import mediapipe as mp
import numpy as np

# Create Dummy 28x28 image
img = np.zeros((28, 28, 3), dtype=np.uint8)
cv2.circle(img, (14,14), 10, (255, 255, 255), -1)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=1
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

res = landmarker.detect(mp_image)
print("Detected:", len(res.hand_landmarks) if hasattr(res, 'hand_landmarks') else 0)
