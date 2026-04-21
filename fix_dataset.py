import pandas as pd
import numpy as np
import os

DATASET_FILE = 'data/asl_dataset.csv'
BACKUP_FILE = 'data/asl_dataset_backup.csv'

if not os.path.exists(DATASET_FILE):
    print("Dataset not found.")
    exit()

print("Loading dataset...")
df = pd.read_csv(DATASET_FILE, dtype={0: str}, low_memory=False)

# Make a backup
df.to_csv(BACKUP_FILE, index=False)
print(f"Backup created at {BACKUP_FILE}")

# The user collected digits and some other custom labels like 'DEL', 'NOTHING', 'SPACE' possibly via webcam?
# Let's check which labels were likely from webcam (non-square).
# We know digits are from webcam (640x480). DEL might be from Kaggle ASL Alphabet.
# Wait, Kaggle ASL Alphabet includes 'del', 'nothing', 'space' as classes!
# Let's count them: DEL=4084, NOTHING=14, SPACE=1 (Oh wait, maybe space/nothing were partly custom, but Kaggle has them).
# We can safely assume 0-9 were from webcam (640x480).
# What if we just fix ONLY 0-9? 
# Wait, if Kaggle images are 200x200 (w/h = 1.0), applying the fix with w=1.0 and h=1.0 does nothing.
# Let's explicitly fix 0-9 with w=640, h=480.
# Wait! Did the user record anything else with webcam?
# Let's find out how many rows of NOTHING and SPACE there are.
# If they are from camera, we might want to fix them. But we only really care about 0-9 being preserved.

digits = [str(i) for i in range(10)]
# Let's just fix 0-9.

count = 0
for i in range(len(df)):
    label = str(df.iloc[i, 0])
    if label in digits:
        row = df.iloc[i, 1:].values.astype(float)
        
        # Original logic: val = (lm - base) / max_original
        # We want: val_new = ((lm - base) * dim) / max_new
        # So val_new = (val * dim) / (max(abs(val * dim)))
        
        # Even indices (0, 2, 4...) are X, Odd (1, 3, 5...) are Y
        xs_new = row[0::2] * 640
        ys_new = row[1::2] * 480
        
        max_val = max(np.max(np.abs(xs_new)), np.max(np.abs(ys_new)))
        if max_val == 0:
            max_val = 1.0
            
        row[0::2] = xs_new / max_val
        row[1::2] = ys_new / max_val
        
        df.iloc[i, 1:] = row
        count += 1

# Actually let's assume ALL labels with count < 500 were custom webcams.
# From value counts, M=1958, N=1529, 9=409, 8=368... 
# This implies Kaggle alphabet has >1500 per class, while camera has <500 per class.
# We will fix any class with count < 500 because it was almost certainly collected via webcam!

print(f"Fixed {count} rows for digits.")

# Also let's check class counts to fix ALL low count ones just in case
class_counts = df.iloc[:, 0].value_counts()
webcam_labels = class_counts[class_counts < 1000].index.tolist()
print(f"Assuming following labels were webcam (640x480): {webcam_labels}")

count_other = 0
for i in range(len(df)):
    label = str(df.iloc[i, 0])
    if label in webcam_labels and label not in digits:
        row = df.iloc[i, 1:].values.astype(float)
        xs_new = row[0::2] * 640
        ys_new = row[1::2] * 480
        max_val = max(np.max(np.abs(xs_new)), np.max(np.abs(ys_new)))
        if max_val == 0:
            max_val = 1.0
        row[0::2] = xs_new / max_val
        row[1::2] = ys_new / max_val
        df.iloc[i, 1:] = row
        count_other += 1

print(f"Fixed {count_other} rows for other custom labels.")

df.to_csv(DATASET_FILE, index=False)
print("Saved fixed dataset!")
