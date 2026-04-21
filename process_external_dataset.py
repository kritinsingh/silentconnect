import os
import cv2
import csv
import mediapipe as mp
import argparse

def process_dataset(dataset_path, start_label=None):
    DATA_DIR = 'data'
    DATASET_FILE = os.path.join(DATA_DIR, 'asl_dataset.csv')
    MODEL_PATH = 'hand_landmarker.task'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    file_exists = os.path.isfile(DATASET_FILE)
    
    f = open(DATASET_FILE, 'a', newline='')
    writer = csv.writer(f)
    
    if not file_exists or os.path.getsize(DATASET_FILE) == 0:
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}']) 
        writer.writerow(header)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    print("Initializing MediaPipe HandLandmarker...")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE, # Using IMAGE mode instead of VIDEO
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3
    )

    try:
        landmarker = HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error creating HandLandmarker: {e}")
        return

    processed_count = 0
    failed_count = 0

    print(f"Scanning directory: {dataset_path}")
    print("This may take a while depending on the size of the dataset...")
    
    for root, dirs, files in os.walk(dataset_path):
        dirs.sort()
        files.sort()
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(root, file)
            
            # Determine Label
            # If the image is inside a subfolder (e.g. 'train/A/001.jpg'), label is 'A'
            if root != dataset_path:
                label = os.path.basename(root).upper()
            else:
                # If images are flat (e.g. 'A_test.jpg'), split by '_' and grab 'A'
                label = file.split('_')[0].upper()
            
            if start_label and label < start_label.upper():
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            try:
                detection_result = landmarker.detect(mp_image)
            except Exception as e:
                print(f"Detection failed for {img_path}: {e}")
                continue
            
            # Draw Points & Save Data
            if detection_result.hand_landmarks:
                # Note: We only save ONE hand per image
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # Apply 1:1 Normalization exactly as we do in live camera
                h, w = image.shape[:2]
                base_px, base_py = hand_landmarks[0].x * w, hand_landmarks[0].y * h
                temp_landmarks = []
                
                for lm in hand_landmarks:
                    temp_landmarks.append((lm.x * w - base_px, lm.y * h - base_py))
                    
                # Max absolute scale for normalization
                max_val = max(max(abs(x), abs(y)) for x, y in temp_landmarks)
                if max_val == 0:
                    max_val = 1.0 # prevent division by zero
                    
                row = [label]
                for x, y in temp_landmarks:
                    row.extend([x / max_val, y / max_val])
                    
                writer.writerow(row)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
            else:
                failed_count += 1
                
    f.close()
    print("------------------------------------------------")
    print(f"Finished processing dataset!")
    print(f"Successfully extracted normalized landmarks: {processed_count} images")
    print(f"Failed to find hand (discarded): {failed_count} images")
    print("------------------------------------------------")
    print("You can now run 'train_classifier.py' to train the AI with this new data.")

if __name__ == '__main__':
    default_path = r"C:\Users\janma\OneDrive\Desktop\asl_alphabet_test"
    
    parser = argparse.ArgumentParser(description="Process image datasets for ASL.")
    parser.add_argument("--path", type=str, default=default_path, help="Path to the dataset directory")
    parser.add_argument("--start-label", type=str, default=None, help="Label to start from (alphabetical)")
    args = parser.parse_args()
    
    process_dataset(args.path, args.start_label)
