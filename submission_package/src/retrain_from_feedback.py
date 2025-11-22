# src/retrain_from_feedback.py
import pandas as pd, os, joblib
from src.preprocess import load_and_process

def merge_and_retrain():
    base = pd.read_csv("data/processed/processed.csv")
    if not os.path.exists("data/feedback/feedback.csv"):
        print("No feedback found")
        return
    fb = pd.read_csv("data/feedback/feedback.csv", names=["transaction","label"])
    fb['text'] = fb['transaction'].apply(lambda s: s.lower())
    merged = pd.concat([base[['text','label']], fb[['text','label']]])
    merged.to_csv("data/processed/merged_for_retrain.csv", index=False)
    # For simplicity, call train script on merged file (modify train script to accept path)
    print("Merged feedback to data/processed/merged_for_retrain.csv")
    # You would then call training pipeline with merged_for_retrain.csv

if __name__=="__main__":
    merge_and_retrain()
