import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque, Counter
from asl_classifier import ASLClassifier
import pyttsx3
from spellchecker import SpellChecker
import threading

def main():
    # --- Configuration ---
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Initialize Core Components
    try:
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        recognizer = GestureRecognizer.create_from_options(options)
    except Exception as e:
        print(f"Failed to create GestureRecognizer: {e}")
        return

    asl_classifier = ASLClassifier()
    
    # Initialize TTS Engine (in a separate thread? No, just init here)
    try:
        engine = pyttsx3.init()
        # Set properties (optional)
        engine.setProperty('rate', 150) 
    except Exception as e:
        print(f"TTS Init Failed: {e}")
        engine = None

    spell = SpellChecker()

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Window
    cv2.namedWindow("ASL Recognizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ASL Recognizer", 1280, 720)

    # State Variables
    start_time = time.time()
    current_word = ""
    recognition_modes = ["All", "Digit", "Character", "Word"]
    current_mode_idx = 0
    current_mode = recognition_modes[current_mode_idx]
    last_added_letter = ""
    last_stable_letter = "..."
    prediction_history = deque(maxlen=10)
    
    REQUIRED_STABILITY = 8 
    WORD_ADD_DELAY = 30 # Frames
    current_hold_count = 0
    
    # UI Colors (Neon / Cyberpunk)
    COLOR_BG = (10, 10, 30) # Dark Blue/Black
    COLOR_ACCENT = (0, 255, 255) # Cyan
    COLOR_TEXT = (255, 255, 255)
    COLOR_HAND = (255, 0, 255) # Magenta
    
    def speak_text(text):
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except:
                pass

    def speak_async(text):
        threading.Thread(target=speak_text, args=(text,), daemon=True).start()

    print("Camera opened. Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        timestamp_ms = int((time.time() - start_time) * 1000)
        recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        raw_letter = "None"
        sign_gesture = "None"
        
        # Create Overlay Layer for Glow Effects
        overlay = img.copy()
        
        if recognition_result.hand_landmarks:
            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                sign_gesture = top_gesture.category_name 

            for hand_landmarks in recognition_result.hand_landmarks:
                # Custom Drawing
                points = []
                for lm in hand_landmarks:
                    px = (int(lm.x * w), int(lm.y * h))
                    points.append(px)
                
                # Draw Connections (Neon Style)
                # mp_hands = mp.solutions.hands  <-- caused AttributeError
                # connections = mp_hands.HAND_CONNECTIONS
                connections = frozenset([
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
                ])
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    cv2.line(overlay, points[start_idx], points[end_idx], COLOR_HAND, 4)
                    cv2.line(img, points[start_idx], points[end_idx], (255, 255, 255), 1) # White core
                
                # Draw Points
                for px in points:
                    cv2.circle(overlay, px, 8, COLOR_ACCENT, -1)
                    cv2.circle(img, px, 4, (255, 255, 255), -1)

                # Bounding Box (Cornered)
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                x_min, y_min = min(x_vals)-30, min(y_vals)-30
                x_max, y_max = max(x_vals)+30, max(y_vals)+30
                
                length = 20
                thick = 4
                cv2.line(overlay, (x_min, y_min), (x_min + length, y_min), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_min, y_min), (x_min, y_min + length), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_max, y_min), (x_max - length, y_min), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_max, y_min), (x_max, y_min + length), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_min, y_max), (x_min + length, y_max), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_min, y_max), (x_min, y_max - length), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_max, y_max), (x_max - length, y_max), COLOR_ACCENT, thick)
                cv2.line(overlay, (x_max, y_max), (x_max, y_max - length), COLOR_ACCENT, thick)
                
                # Classify
                handedness_label = "Right" # Default
                if recognition_result.handedness:
                     if len(recognition_result.handedness) > 0:
                         handedness_label = recognition_result.handedness[0][0].category_name
                
                # Classify
                raw_letter = asl_classifier.classify(hand_landmarks, w, h, handedness_label)
        
                if raw_letter == "Uninitialized":
                     cv2.putText(img, "MODEL NOT TRAINED", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                     cv2.putText(img, "Run 'collect_data.py' then 'train_classifier.py'", (w//2 - 350, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Blend Overlay
        alpha = 0.6
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Smoothing & Logic
        if current_mode == "Digit" and raw_letter == "V":
            raw_letter = "2"
        elif current_mode in ["Character", "Word"] and raw_letter == "2":
            raw_letter = "V"
            
        is_valid = False
        if raw_letter not in ["Unknown", "None", "Uninitialized", "Error"]:
             if current_mode == "All":
                 is_valid = True
             elif current_mode == "Digit" and raw_letter.isdigit():
                 is_valid = True
             elif current_mode == "Character" and len(raw_letter) == 1 and raw_letter.isalpha():
                 is_valid = True
             elif current_mode == "Word" and len(raw_letter) > 1 and raw_letter.isalpha():
                 is_valid = True
             
             if "Space" in raw_letter:
                 is_valid = True 

        if is_valid:
            prediction_history.append(raw_letter)
        else:
             if sign_gesture != "None":
                 # Maybe map gestures to functional keys if needed
                 pass
             prediction_history.append(None)

        current_display_letter = "..."
        valid_preds = [p for p in prediction_history if p is not None]
        
        if valid_preds:
            most_common, count = Counter(valid_preds).most_common(1)[0]
            if count >= REQUIRED_STABILITY:
                current_display_letter = most_common
                last_stable_letter = most_common
            else:
                last_stable_letter = "..."
        else:
            last_stable_letter = "..."

        # Word Formation
        if last_stable_letter != "...":
             current_hold_count += 1
             display_color = (0, 255, 0) # Green for stable
             if current_hold_count > WORD_ADD_DELAY:
                  if last_stable_letter != last_added_letter:
                       # Handle Special Keys
                       char_to_add = last_stable_letter.split(" / ")[0]
                       
                       if "Space" in last_stable_letter:
                            current_word += " "
                       elif "Delete" in sign_gesture: # Gesture override
                            current_word = current_word[:-1]
                       else:
                            current_word += char_to_add
                            
                       last_added_letter = last_stable_letter
                       current_hold_count = 0 
                       
                       # Feedback Flash
                       cv2.rectangle(img, (0, 0), (w, h), (0, 255, 0), 20)
        else:
             current_hold_count = 0
             display_color = COLOR_TEXT

        # UI Layout Helpers
        def draw_text(image, text, pos, font_scale, color, thickness=2):
            cv2.putText(image, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,0,0), thickness+1)
            cv2.putText(image, text, pos, cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)

        # Transparent Overlays
        overlay_ui = img.copy()
        # Top Header (Reduced size)
        cv2.rectangle(overlay_ui, (0, 0), (w, 60), (15, 15, 15), -1)
        cv2.rectangle(overlay_ui, (0, 60), (w, 62), COLOR_ACCENT, -1)
        # Bottom Text Area (Reduced size)
        cv2.rectangle(overlay_ui, (0, h-60), (w, h), (20, 20, 25), -1)
        cv2.rectangle(overlay_ui, (0, h-60), (w, h-58), COLOR_ACCENT, -1)
        
        alpha_ui = 0.8
        img = cv2.addWeighted(overlay_ui, alpha_ui, img, 1 - alpha_ui, 0)

        # Header Text (Removed ASL DECODER)
        draw_text(img, f"MODE: [{current_mode}]", (w - 300, 30), 0.6, (0, 255, 0), 2)
        draw_text(img, "Press 'm' to change", (w - 300, 50), 0.4, (200, 200, 200), 1)

        # Current Letter Box (Center) (Reduced size)
        cx, cy = w // 2, 90
        cv2.circle(img, (cx, cy), 45, (20, 20, 20), -1)
        cv2.circle(img, (cx, cy), 45, display_color, 3)
        cv2.circle(img, (cx, cy), 38, display_color, 1)
        
        text_size = cv2.getTextSize(last_stable_letter, cv2.FONT_HERSHEY_DUPLEX, 1.8, 3)[0]
        cv2.putText(img, last_stable_letter, (cx - text_size[0]//2, cy + text_size[1]//2 - 5), cv2.FONT_HERSHEY_DUPLEX, 1.8, display_color, 3)

        # Autocorrect Suggestion (Above Footer)
        if current_word.strip():
            last_word = current_word.split(" ")[-1]
            if last_word:
                correction = spell.correction(last_word)
                if correction and correction != last_word:
                    draw_text(img, f"Suggestion: {correction} (?)", (30, h-75), 0.7, (100, 255, 255), 2)

        # Bottom Text Area
        cursor = "_" if (int(time.time() * 2) % 2) == 0 else " "
        draw_text(img, current_word + cursor, (40, h-20), 1.0, COLOR_TEXT, 2)

        # Speak Button (Virtual) / Gesture
        # If "Thumb_Up" gesture is recognized, speak the word
        if sign_gesture == "Thumb_Up":
             speak_async(current_word)
             sign_gesture = "None" # Debounce
             time.sleep(1.0)
             current_word = "" # Clear after speaking? Optional.
        
        cv2.imshow("ASL Recognizer", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_mode_idx = (current_mode_idx + 1) % len(recognition_modes)
            current_mode = recognition_modes[current_mode_idx]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
