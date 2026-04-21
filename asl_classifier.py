import pickle
import numpy as np
import os

class ASLClassifier:
    def __init__(self):
        self.model_path = 'model.p'
        self.model = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"Warning: Model file '{self.model_path}' not found. Classifier will return 'Uninitialized'.")

    def classify(self, landmarks, w=1.0, h=1.0, handedness="Right"):
        """
        Classifies the hand landmarks using the trained model.
        landmarks: List of landmarks (objects with x, y attributes).
        w, h: Width and height of the image to ensure aspect-ratio invariant features.
        handedness: "Right" or "Left" (Not used in this ML model version, but kept for interface).
        Returns: String (Letter/Number) or "Unknown"/"Uninitialized"
        """
        if self.model is None:
            # Try reloading in case it was created after init
            self.load_model()
            if self.model is None:
                return "Uninitialized"

        data_aux = []

        # 1. Translate to wrist
        base_px, base_py = landmarks[0].x * w, landmarks[0].y * h
        temp_landmarks = []
        
        for lm in landmarks:
            temp_landmarks.append((lm.x * w - base_px, lm.y * h - base_py))
            
        # 2. Scale by absolute max distance
        max_val = max(max(abs(x), abs(y)) for x, y in temp_landmarks)
        if max_val == 0:
            max_val = 1.0 # prevent division by zero
            
        # Normalize and flatten
        for x, y in temp_landmarks:
            data_aux.extend([x / max_val, y / max_val])

        # Prediction
        try:
             prediction = self.model.predict([np.asarray(data_aux)])
             return prediction[0]
        except Exception as e:
             # print(f"Prediction Error: {e}")
             return "Error"

