# Automated AI-Based Financial Transaction Categorisation

This project implements an end to end, fully local AI system for automatically categorizing financial transactions using machine learning, configurable taxonomies, explainable AI techniques, and human feedback integration. It is designed to provide a cost-effective, privacy-focused alternative to third party categorization APIs.

## 1. Project Overview

This project solves these challenges by providing a complete, standalone ML-powered categorisation engine that runs entirely on local infrastructure, supports custom categories, integrates explainability, and continuously improves through user feedback.

Modern financial applications require fast and accurate categorisation of raw transaction text such as “Starbucks”, “Amazon Purchase”, or “Shell Gas”. Traditional approaches rely on external paid APIs, which introduce cost, latency, and data-privacy concerns.

The system includes:

•	Machine Learning Model (Linear SVM + TF-IDF)

•	Fully local inference (no external APIs)

•	Customisable taxonomy using YAML

•	Explainability using model coefficients

•	Human feedback loop with CSV storage

•	Web interface built with Flask

## 2. Technology Stack

•	Python 3.10 – The core programming language used to implement the entire machine learning pipeline and web application.

•	Scikit-learn – Provides the ML algorithms and tools for training, evaluating, and deploying the transaction categorization model.

•	TF-IDF Vectorizer – Converts raw transaction text into numerical feature vectors used by the machine learning model.

•	Linear Support Vector Machine (SVM) – The main classification algorithm that predicts transaction categories with high accuracy.

•	Pandas & NumPy – Used for data processing, cleaning, transformation, and dataset manipulation.

•	Flask – Powers the local web interface for running predictions, showing explanations, and collecting user feedback.

•	YAML – Stores the editable taxonomy configuration, allowing categories to be updated without modifying code.

•	Matplotlib – Generates evaluation artifacts including the confusion matrix for model assessment.

•	Chart.js – Renders feature contribution and explainability charts in the browser UI.

•	HTML/CSS/Bootstrap – Builds the responsive and modern frontend layout for the user interface.


## 3. System Architecture

### 1. Dataset Generation and Ingestion
Collects raw or synthetic transaction strings and stores them locally to build a customizable training dataset.

### 2. Preprocessing & Vectorization 
Cleans input text and converts it into TF-IDF numerical vectors that the ML model can understand.

### 3. Model Training 
Uses a Linear SVM classifier to learn patterns and boundaries between transaction categories.

### 4. Evaluation 
Produces confusion matrices and F1-score reports to measure classification performance and ensure transparency.

### 5. Explainability 
Extracts feature importance signals from model coefficients to show why a prediction was made.

### 6. Inference API 
A Flask-based prediction endpoint that returns real-time category and confidence scores.

### 7. Local Web UI 
A browser interface enabling users to categorize transactions, view explanations, and interact with the system.

### 8. Human Feedback Loop
Stores user-corrected labels into feedback.csv to continuously refine and improve the model.

### 9. Retraining Capability 
Allows new training sessions that incorporate user feedback, enabling adaptive and ever-improving performance.

## 4. Data Model & Storage

•	Processed Dataset – The cleaned and vector-ready transaction dataset is stored at data/processed/processed.csv for training and evaluation.

•	Feedback Storage – User-submitted corrections are logged in data/feedback/feedback.csv to enable continuous model improvement.

•	Taxonomy Definition – All category names and IDs are defined in configs/taxonomy.yaml, allowing full admin control without code changes.

•	Model Artifacts – Trained ML models and vectorizers are saved in artifacts/models/ for reproducible inference and retraining.

•	Evaluation Artifacts – Metrics such as confusion matrices and classification reports are stored in artifacts/metrics/ for transparent performance analysis.

## 5. AI / ML Components 

### 1. Vectorization: TF-IDF

Converts raw transaction text into numerical feature vectors by capturing word importance across the dataset.

### 2. Model: Linear Support Vector Machine

A fast and highly accurate linear classifier used to predict the correct transaction category based on vectorized features.

### 3. Performance (Macro F1 Score: 0.9969, Accuracy: 0.9969)

Demonstrates business grade classification performance on the evaluation dataset, exceeding the required 0.90 macro F1 target.

### 4. Evaluation Artifacts: confusion_matrix.png

Provides a visual representation of true vs. predicted categories for transparency and error analysis.

### 5. Evaluation Artifacts: classification_report.json

Stores detailed per class precision, recall, and F1 metrics for reproducible evaluation and auditing.

