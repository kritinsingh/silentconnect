import pickle
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv('data/asl_dataset.csv', dtype={0: str}, low_memory=False)
class_counts = df.iloc[:, 0].value_counts()
valid_classes = class_counts[class_counts >= 5].index
df = df[df.iloc[:, 0].isin(valid_classes)]

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].astype(str).values

with open('model.p', 'rb') as f:
    model = pickle.load(f)['model']

y_pred = model.predict(x)
print(classification_report(y, y_pred))
