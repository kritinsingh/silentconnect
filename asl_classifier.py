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

    def classify(self, landmarks, handedness="Right"):
        """
        Classifies the hand landmarks using the trained model.
        landmarks: List of landmarks (objects with x, y attributes).
        handedness: "Right" or "Left" (Not used in this ML model version, but kept for interface).
        Returns: String (Letter/Number) or "Unknown"/"Uninitialized"
        """
        if self.model is None:
            # Try reloading in case it was created after init
            self.load_model()
            if self.model is None:
                return "Uninitialized"

        data_aux = []
        x_ = []
        y_ = []

        for lm in landmarks:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in landmarks:
             # Normalize relative to min/max? 
             # The training data was collected as RAW (x, y). 
             # Wait, in collect_data.py I saved RAW (x, y).
             # Efficient models usually benefit from normalization, 
             # but if I trained on raw, I must predict on raw.
             # Actually, raw coordinates from MediaPipe are already normalized [0, 1].
             # So they are fine as is, provided the camera aspect ratio doesn't distort them heavily.
             # Let's match whatever collect_data.py did.
             
             # collect_data.py did: landmark_list.extend([lm.x, lm.y])
             data_aux.extend([lm.x, lm.y])

        # Prediction
        try:
             prediction = self.model.predict([np.asarray(data_aux)])
             return prediction[0]
        except Exception as e:
             # print(f"Prediction Error: {e}")
             return "Error"

