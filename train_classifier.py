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
        df = pd.read_csv(DATASET_FILE, dtype={0: str}, low_memory=False)
    except pd.errors.EmptyDataError:
        print("Error: Dataset is empty.")
        return

    if df.empty or len(df) < 5:
        print("Error: Not enough data to train.")
        return

    # Filter out classes with too few samples to allow stratification
    class_counts = df.iloc[:, 0].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df.iloc[:, 0].isin(valid_classes)]

    if df.empty or len(df) < 5:
        print("Error: Not enough data after filtering rare classes.")
        return

    x = df.iloc[:, 1:].values # Features (landmarks)
    y = df.iloc[:, 0].astype(str).values  # Labels (Ensure strictly strings to prevent numpy sort crashes)

    # --- Data Augmentation ---
    print(f"Original dataset size: {len(x)}")
    
    # 1. Flip X-coordinates to simulate opposite hand (Left vs Right)
    x_flipped = x.copy()
    x_flipped[:, 0::2] = -x_flipped[:, 0::2] # negate all X coordinates
    
    # 2. Add small random noise to simulate different hand sizes/proportions
    def add_noise(data, noise_level=0.03):
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    # Create variations: Original, Flipped, and noisy versions of both
    x_augmented = np.vstack((
        x,
        x_flipped,
        add_noise(x, 0.02),
        add_noise(x_flipped, 0.02),
        add_noise(x, 0.04),
        add_noise(x_flipped, 0.04)
    ))
    
    y_augmented = np.concatenate((y, y, y, y, y, y))
    print(f"Augmented dataset size: {len(x_augmented)}")

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(x_augmented, y_augmented, test_size=0.2, shuffle=True, stratify=y_augmented)

    # Train Model (Optimized to prevent massive file sizes & train fast)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', min_samples_leaf=2, max_depth=50, n_jobs=-1)
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
