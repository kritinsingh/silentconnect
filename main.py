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
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
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
                mp_hands = mp.solutions.hands
                connections = mp_hands.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    cv2.line(overlay, points[start_idx], points[end_idx], COLOR_HAND, 4)
                    cv2.line(img, points[start_idx], points[end_idx], (255, 255, 255), 1) # White core
                
                # Draw Points
                for px in points:
                    cv2.circle(overlay, px, 8, COLOR_ACCENT, -1)
                    cv2.circle(img, px, 4, (255, 255, 255), -1)

                # Bounding Box
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                bbox = (min(x_vals)-20, min(y_vals)-20, max(x_vals)+20, max(y_vals)+20)
                cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLOR_ACCENT, 2)
                
                # Classify
                handedness_label = "Right" # Default
                if recognition_result.handedness:
                     if len(recognition_result.handedness) > 0:
                         handedness_label = recognition_result.handedness[0][0].category_name
                
        raw_letter = asl_classifier.classify(hand_landmarks, handedness_label)
        
        if raw_letter == "Uninitialized":
             cv2.putText(img, "MODEL NOT TRAINED", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
             cv2.putText(img, "Run 'collect_data.py' then 'train_classifier.py'", (w//2 - 350, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Blend Overlay
        alpha = 0.6
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Smoothing & Logic
        if raw_letter not in ["Unknown", "None", "Uninitialized", "Error"]:
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

        # UI Layout
        # Top Header
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(img, "ASL DECODER", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_ACCENT, 2)
        
        # Current Letter Box
        cv2.circle(img, (w//2, 100), 60, (0, 0, 0), -1)
        cv2.circle(img, (w//2, 100), 55, display_color, 2)
        text_size = cv2.getTextSize(last_stable_letter, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        cv2.putText(img, last_stable_letter, (w//2 - text_size[0]//2, 115), cv2.FONT_HERSHEY_SIMPLEX, 2, display_color, 3)

        # Bottom Text Area
        cv2.rectangle(img, (0, h-100), (w, h), (20, 20, 20), -1)
        cv2.putText(img, current_word + "|", (30, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT, 2)

        # Autocorrect Suggestion (Top Right)
        if current_word.strip():
            last_word = current_word.split(" ")[-1]
            if last_word:
                correction = spell.correction(last_word)
                if correction and correction != last_word:
                    cv2.putText(img, f"Did you mean: {correction}?", (w-400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                    
                    # Auto-replace on specific gesture? 
                    # For now just show suggestion.

        # Speak Button (Virtual) / Gesture
        # If "Thumb_Up" gesture is recognized, speak the word
        if sign_gesture == "Thumb_Up":
             speak_async(current_word)
             sign_gesture = "None" # Debounce
             time.sleep(1.0)
             current_word = "" # Clear after speaking? Optional.
        
        cv2.imshow("ASL Recognizer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
