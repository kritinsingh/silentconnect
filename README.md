# ASL Hand Gesture Recognition & Text-to-Speech App

This application uses computer vision and machine learning interpret American Sign Language (ASL) hand gestures in real-time. It translates detected gestures into text, forms words, provides spell-checking suggestions, and can speak the formed sentences using Text-to-Speech (TTS).

## Features

*   **Real-time Hand Tracking**: Uses MediaPipe for robust hand landmark detection.
*   **Custom Gesture Classification**: Train your own ASL gestures (A-Z, 0-9, static signs) using a Random Forest Classifier.
*   **Text Construction**: Recognized characters are assembled into words with stability checks to prevent flickering.
*   **Text-to-Speech (TTS)**: Converts the formed text into spoken audio.
*   **Spell Checking**: Suggests corrections for the current word.
*   **Interactive UI**: Visual feedback for tracking, recognition confidence, and mode status.

## Project Structure

*   `collect_data.py`: Script to capture hand landmark data from your webcam and save it to a CSV dataset.
*   `train_classifier.py`: Script to train a machine learning model (Random Forest) on the collected data.
*   `asl_classifier.py`: Helper class to load the trained model and predict gestures in real-time.
*   `main.py`: The main application that runs the webcam loop, performs recognition, and handles the UI/TTS.
*   `data/`: Directory where the dataset (`asl_dataset.csv`) is stored.
*   `model.p`: The trained model file (generated after running `train_classifier.py`).

## Installation

1.  **Clone the repository** (if applicable) or download the source code.

2.  **Install Dependencies**:
    You need Python installed. Run the following command to install the required libraries:

    ```bash
    pip install opencv-python mediapipe numpy pandas scikit-learn pyttsx3 pyspellchecker
    ```

## Usage

### 1. Collect Data (`collect_data.py`)

Before the app can recognize your specific signs, you need to collect training data.

1.  Run the script:
    ```bash
    python collect_data.py
    ```
2.  Enter the label you want to record (e.g., 'A', 'B', 'Hello').
3.  The camera window will open.
    *   **Press 's'**: To save a single frame (good for static poses).
    *   **Press 'r'**: To toggle continuous recording (saves frames automatically while held).
    *   **Press 'q'**: To quit.
4.  Repeat this for all the gestures you want to recognize. Aim for at least 50-100 samples per gesture for better accuracy.

### 2. Train the Model (`train_classifier.py`)

Once you have collected data for all your gestures:

1.  Run the training script:
    ```bash
    python train_classifier.py
    ```
2.  This will read the `data/asl_dataset.csv`, train a Random Forest classifier, and save the model to `model.p`.
3.  Validation accuracy will be printed to the console.

### 3. Run the Application (`main.py`)

Now you can run the main recognition app:

1.  Start the app:
    ```bash
    python main.py
    ```
2.  **Usage**:
    *   Show your hand to the camera.
    *   The recognized character/gesture will appear on the screen.
    *   **Hold a gesture** to "type" it (letters are added after a short delay of stability).
    *   **Special Gestures**:
        *   If you trained a `Space` gesture, it adds a space.
        *   If you trained a `Delete` gesture, it removes the last character.
        *   **Thumb Up**: Triggers the Text-to-Speech engine to speak the current sentence.
    *   **Press 'q'** to exit.

## Troubleshooting

*   **Camera not opening**: Ensure no other app is using the webcam.
*   **"Model not found"**: Make sure you ran `train_classifier.py` successfully and `model.p` exists.
*   **Low Accuracy**: Collect more data with varied angles, distances, and lighting conditions.
*   **TTS not working**: Ensure your system has a working text-to-speech engine (SAPI5 on Windows, nsss on Mac, espeak on Linux).

## Credits

Built using [MediaPipe](https://developers.google.com/mediapipe), [Scikit-Learn](https://scikit-learn.org/), and [OpenCV](https://opencv.org/).
