# ASL Recognition Project Summary

This document provides a comprehensive overview of the current state of the American Sign Language (ASL) Recognition project. It is intended to help an LLM understand the codebase and functionality.

## Project Overview
This project is a real-time computer vision application that recognizes ASL hand gestures using a webcam. It translates these gestures into text, assembles words, and provides text-to-speech (TTS) output.

## Technology Stack
- **Language:** Python
- **Computer Vision:** OpenCV (`cv2`), MediaPipe (`mediapipe`)
- **Machine Learning:** Scikit-learn (`sklearn`) - Uses a Random Forest Classifier.
- **Text-to-Speech:** `pyttsx3`
- **Autocorrect:** `pyspellchecker`
- **Data Handling:** `pandas`, `numpy`, `csv`, `pickle`

## Project Structure & Files

### 1. `collect_data.py` (Data Collection)
This script is used to create the training dataset.
- **Functionality**:
    - Initializes the webcam and MediaPipe Hand Landmarker.
    - Asks the user for a target label (e.g., 'A', 'B', 'Hello').
    - Captures frames from the webcam.
    - Extracts 21 hand landmarks (x, y coordinates) for each frame.
    - Saves the landmarks and the label to `data/asl_dataset.csv`.
- **Key Features**:
    - Press 'R' to toggle continuous recording.
    - Press 'S' to save a single frame.
    - Press 'Q' to quit.

### 2. `train_classifier.py` (Model Training)
This script trains the machine learning model using the collected data.
- **Functionality**:
    - Loads the dataset from `data/asl_dataset.csv`.
    - Splits the data into training and testing sets (80/20 split).
    - Trains a **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`).
    - Evaluates the model accuracy on the test set.
    - Saves the trained model to `model.p` using `pickle`.

### 3. `asl_classifier.py` (Inference Wrapper)
This is a helper class to load and use the trained model.
- **Functionality**:
    - Loads the `model.p` file upon initialization.
    - Provides a `classify(landmarks)` method.
    - Takes raw landmarks from MediaPipe, formats them to match training data, and returns the predicted character/label.
    - Handles cases where the model is not yet trained/found (returns "Uninitialized").

### 4. `main.py` (Main Application)
This is the main entry point for the real-time recognition application.
- **Core Loop**:
    - Captures video from the webcam.
    - Uses MediaPipe to detect hands and extract landmarks.
    - Draws a "Neon/Cyberpunk" style overlay on the hand (skeleton and joints).
    - Passes landmarks to `ASLClassifier` to get a prediction.
- **Word Formation Logic**:
    - Buffers predictions to ensure stability (must be the same gesture for ~8 frames).
    - Appends stable characters to a word string.
    - Handles "Space" gesture to separate words.
    - Handles "Delete" gesture (if implemented/trained) or backspace logic.
    - Uses a timer to prevent rapid-fire duplicate letters.
- **Key Features**:
    - **Text-to-Speech**: Speaks the current word when a specific gesture (e.g., "Thumb_Up") is recognized (via `speak_async`).
    - **Autocorrect**: Suggests corrections for the current word using `pyspellchecker` and displays them on screen.
    - **UI**: Detailed overlay with separate bounding box, connection lines, and a text display area.

## Data Flow
1.  **Input**: Webcam Video Feed.
2.  **Processing**: MediaPipe extracts (x, y) coordinates of 21 hand landmarks.
3.  **Classification**: Random Forest Model predicts the Class (Letter/Gesture) based on landmark coordinates.
4.  **Logic**: Smoothing algorithm filters noise; State machine handles word construction.
5.  **Output**: Visual feedback on screen + Audio feedback via TTS.

## Setup & Usage
1.  **Install Dependencies**: `pip install opencv-python mediapipe scikit-learn pandas pyttsx3 pyspellchecker`
2.  **Collect Data**: Run `python collect_data.py`, enter a label (e.g., 'A'), and record samples. Repeat for all desired gestures.
3.  **Train Model**: Run `python train_classifier.py` to generate `model.p`.
4.  **Run App**: Execute `python main.py` to start the recognizer.
