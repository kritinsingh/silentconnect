import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_model():
    DATA_DIR = 'data'
    DATASET_FILE = os.path.join(DATA_DIR, 'asl_dataset.csv')
    MODEL_FILE = 'model.p'

    if not os.path.exists(DATASET_FILE):
        print(f"Error: Dataset not found at {DATASET_FILE}")
        return

    # Load Data
    try:
        df = pd.read_csv(DATASET_FILE)
    except pd.errors.EmptyDataError:
        print("Error: Dataset is empty.")
        return

    if df.empty or len(df) < 5:
        print("Error: Not enough data to train.")
        return

    x = df.iloc[:, 1:].values # Features (landmarks)
    y = df.iloc[:, 0].values  # Labels

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    # Train Model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Evaluate
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    print(f'{score * 100}% of samples were classified correctly !')

    # Save Model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model}, f)

    print(f"Model saved to {MODEL_FILE}")

if __name__ == '__main__':
    train_model()
