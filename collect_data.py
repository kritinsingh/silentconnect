import cv2
import mediapipe as mp
import csv
import os
import time

def collect_data():
    # --- Configuration ---
    DATA_DIR = 'data'
    DATASET_FILE = os.path.join(DATA_DIR, 'asl_dataset.csv')
    MODEL_PATH = 'hand_landmarker.task'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Check if dataset exists to write header
    file_exists = os.path.isfile(DATASET_FILE)
    
    # Open CSV in append mode
    f = open(DATASET_FILE, 'a', newline='')
    writer = csv.writer(f)
    
    # Write header if new file
    if not file_exists:
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}']) 
        writer.writerow(header)

    # --- MediaPipe Tasks Setup ---
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        landmarker = HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error creating HandLandmarker: {e}")
        print(f"Ensure '{MODEL_PATH}' exists in the current directory.")
        return

    cap = cv2.VideoCapture(0)
    
    print("--------------------------------------------------")
    print("ASL Data Collector (Tasks API)")
    print("--------------------------------------------------")
    print("Enter the label you want to collect data for (e.g. 'A', 'Hello').")
    target_label = input("Target Label: ").strip().upper()
    print(f"\nCollecting data for: '{target_label}'")
    print("Press 's' to save a single frame.")
    print("Press 'r' to toggle continuous recording.")
    print("Press 'q' to quit.")
    print("--------------------------------------------------")

    count = 0
    is_recording = False
    start_time = time.time()
    
    # Define connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8), # Index
        (5, 9), (9, 10), (10, 11), (11, 12), # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17) # Wrist to Pinky Base
    ]

    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Draw Landmarks (Manual)
        frame_has_hand = False
        current_landmarks_list = []

        if detection_result.hand_landmarks:
            frame_has_hand = True
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, c = image.shape
                
                # Draw Connections
                for p1_idx, p2_idx in HAND_CONNECTIONS:
                    p1 = hand_landmarks[p1_idx]
                    p1_px = (int(p1.x * w), int(p1.y * h))
                    p2 = hand_landmarks[p2_idx]
                    p2_px = (int(p2.x * w), int(p2.y * h))
                    cv2.line(image, p1_px, p2_px, (200, 200, 200), 2)

                # Draw Points & Save Data
                # Note: We only save ONE hand per frame (the first one)
                if not current_landmarks_list:
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
                        current_landmarks_list.extend([lm.x, lm.y])

        # UI Overlay
        status_color = (0, 255, 0) if not is_recording else (0, 0, 255)
        status_text = "Recording..." if is_recording else "Paused"
        
        cv2.putText(image, f"Label: {target_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Count: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Status: {status_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow('ASL Data Collector', image)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            print(f"Recording: {is_recording}")
        elif key == ord('s'):
             # Manual trigger (logic handled below)
             pass 
        
        # Save Logic
        should_save = False
        if key == ord('s') and frame_has_hand:
            should_save = True
        elif is_recording and frame_has_hand:
            should_save = True
            
        if should_save and current_landmarks_list:
             row = [target_label] + current_landmarks_list
             writer.writerow(row)
             f.flush() # Ensure it's written immediately
             count += 1
             if count % 10 == 0:
                  print(f"Collected {count} samples for {target_label}")
             
    cap.release()
    f.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()
