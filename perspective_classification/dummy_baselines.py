import random
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from utils import DATA_DIR

def info_baseline(data):
    y_true = data["s_perspective_1"]
    labels = y_true.unique().tolist()
    y_pred = ["information"] * len(y_true)
    print_metrics(labels, y_true, y_pred)

def random_baseline(data):
    y_true = data["s_perspective_1"]
    labels = y_true.unique().tolist()
    y_pred = random.choices(labels, k=len(y_true))
    print_metrics(labels, y_true, y_pred)

def print_metrics(labels, y_true, y_pred):
    acc = sum([1 for y1, y2 in zip(y_true, y_pred) if y1 == y2]) / len(y_true)

    print(labels)
    print(f"Accuracy: {acc:.3f}")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(f1_score(y_true, y_pred, labels=labels, average=None))
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted")
    print(f"Weighted F1: {weighted_f1:.3f}")
    print()

if __name__ == "__main__":
    file = DATA_DIR / "openai_test_data_11_10_info_merged_classified_mod.csv"
    data = pd.read_csv(file)
    random_baseline(data)
    info_baseline(data)
    