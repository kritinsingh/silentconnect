import os
import pandas as pd
import numpy as np
import cv2

DATASET_FILE = 'data/asl_dataset.csv'
OUTPUT_DIR = 'Dataset_Visualization'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading dataset...")
df = pd.read_csv(DATASET_FILE, dtype={0: str}, low_memory=False)

# Filter valid classes like in train_classifier
class_counts = df.iloc[:, 0].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df.iloc[:, 0].isin(valid_classes)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17) # Wrist to Pinky Base
]

labels = np.sort(df.iloc[:, 0].unique())

# Number of samples per class to save
SAMPLES_PER_CLASS = 5

IMAGE_SIZE = 400
SCALE = 150

for label in labels:
    # Create subfolder for each label
    label_dir = os.path.join(OUTPUT_DIR, str(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        
    subset = df[df.iloc[:, 0] == label]
    # take up to SAMPLES_PER_CLASS random samples
    subset = subset.sample(n=min(SAMPLES_PER_CLASS, len(subset)))
    
    for idx, row in enumerate(subset.itertuples(index=False)):
        features = row[1:] # 42 values
        
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        
        # We want to draw on a white background or dark background. Let's do dark blue.
        img[:] = (20, 20, 40)
        
        points = []
        for i in range(21):
            x = features[i * 2]
            y = features[i * 2 + 1]
            
            # Convert normalized space [-1.0, 1.0] to image space [0, 400]
            # Wrist is (0,0) in normalized space. Let's put wrist at (200, 350)
            px = int(float(x) * SCALE + IMAGE_SIZE // 2)
            py = int(float(y) * SCALE + 300)
            points.append((px, py))
            
        # Draw Connections
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            cv2.line(img, points[p1_idx], points[p2_idx], (0, 255, 255), 3) # Yellow connections
            
        # Draw Points
        for px, py in points:
            cv2.circle(img, (px, py), 6, (255, 0, 255), -1) # Magenta joints
            cv2.circle(img, (px, py), 3, (255, 255, 255), -1) # White center
            
        # Draw Label Text
        cv2.putText(img, f"Class: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        filename = f"{label}_sample_{idx+1}.png"
        filepath = os.path.join(label_dir, filename)
        cv2.imwrite(filepath, img)

print(f"Created visualization dataset in folder: {os.path.abspath(OUTPUT_DIR)}")
