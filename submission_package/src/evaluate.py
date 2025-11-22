# src/evaluate.py
import joblib, pandas as pd, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
import os
from src.preprocess import load_and_process

def evaluate(model_path="artifacts/checkpoints/baseline.joblib"):
    df = load_and_process()
    model = joblib.load(model_path)
    X = df['text']; y = df['label']
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, digits=4))
    labels = model.classes_
    cm = confusion_matrix(y, y_pred, labels=labels)
    os.makedirs("artifacts/metrics", exist_ok=True)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.savefig("artifacts/metrics/confusion_matrix.png")
    print("Saved confusion matrix to artifacts/metrics/confusion_matrix.png")

if __name__=="__main__":
    evaluate()
