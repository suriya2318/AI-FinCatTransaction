# Automated AI-Based Financial Transaction Categorisation

This project implements an end to end, fully local AI system for automatically categorizing financial transactions using machine learning, configurable taxonomies, explainable AI techniques, and human feedback integration. It is designed to provide a cost-effective, privacy-focused alternative to third party categorization APIs.

## 1. Project Overview

This project solves these challenges by providing a complete, standalone ML-powered categorisation engine that runs entirely on local infrastructure, supports custom categories, integrates explainability, and continuously improves through user feedback.

Modern financial applications require fast and accurate categorisation of raw transaction text such as “Starbucks”, “Amazon Purchase”, or “Shell Gas”. Traditional approaches rely on external paid APIs, which introduce cost, latency, and data-privacy concerns.

The system includes:

• Machine Learning Model (Linear SVM + TF-IDF)

• Fully local inference (no external APIs)

• Customisable taxonomy using YAML

• Explainability using model coefficients

• Human feedback loop with CSV storage

• Web interface built with Flask

## 2. Technology Stack

• Python 3.10 – The core programming language used to implement the entire machine learning pipeline and web application.

• Scikit-learn – Provides the ML algorithms and tools for training, evaluating, and deploying the transaction categorization model.

• TF-IDF Vectorizer – Converts raw transaction text into numerical feature vectors used by the machine learning model.

• Linear Support Vector Machine (SVM) – The main classification algorithm that predicts transaction categories with high accuracy.

• Pandas & NumPy – Used for data processing, cleaning, transformation, and dataset manipulation.

• Flask – Powers the local web interface for running predictions, showing explanations, and collecting user feedback.

• YAML – Stores the editable taxonomy configuration, allowing categories to be updated without modifying code.

• Matplotlib – Generates evaluation artifacts including the confusion matrix for model assessment.

• Chart.js – Renders feature contribution and explainability charts in the browser UI.

• HTML/CSS/Bootstrap – Builds the responsive and modern frontend layout for the user interface.

## 3. System Architecture

• Dataset Generation and Ingestion - Collects raw or synthetic transaction strings and stores them locally to build a customizable training dataset.

• Preprocessing & Vectorization - Cleans input text and converts it into TF-IDF numerical vectors that the ML model can understand.

• Model Training - Uses a Linear SVM classifier to learn patterns and boundaries between transaction categories.

• Evaluation - Produces confusion matrices and F1-score reports to measure classification performance and ensure transparency.

• Explainability - Extracts feature importance signals from model coefficients to show why a prediction was made.

• Inference API - A Flask-based prediction endpoint that returns real time category and confidence scores.

• Local Web UI - A browser interface enabling users to categorize transactions, view explanations, and interact with the system.

• Human Feedback Loop - Stores user corrected labels into feedback.csv to continuously refine and improve the model.

• Retraining Capability - Allows new training sessions that incorporate user feedback, enabling adaptive and ever improving performance.

## 4. Data Model & Storage

• Processed Dataset – The cleaned and vector-ready transaction dataset is stored at data/processed/processed.csv for training and evaluation.

• Feedback Storage – User-submitted corrections are logged in data/feedback/feedback.csv to enable continuous model improvement.

• Taxonomy Definition – All category names and IDs are defined in configs/taxonomy.yaml, allowing full admin control without code changes.

• Model Artifacts – Trained ML models and vectorizers are saved in artifacts/models/ for reproducible inference and retraining.

• Evaluation Artifacts – Metrics such as confusion matrices and classification reports are stored in artifacts/metrics/ for transparent performance analysis.

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

## 6. Security and Compliance

• 100% local processing - All model operations run entirely on the user's device, ensuring full data ownership and eliminating external dependencies.

• No external APIs - The system does not send any transaction data to third party services, preventing API related privacy or security risks.

• No PII collected - The model handles generic transaction descriptions without storing or processing personally identifiable information.

• User feedback stored locally - Corrections provided by users are saved in a secure local CSV file, maintaining full control over learning data.

• Transparent explainability methods - Feature based explanations offer clear visibility into how each classification decision is made, supporting responsible AI practices.

## 7. Scalability and Performance

• Inference < 50ms locally - The model delivers near instant category predictions, enabling smooth real time usage even on modest hardware.

• Model size < 10 MB - A compact model footprint ensures fast loading, low memory usage, and easy deployment in local or embedded environments.

• Supports batch inference - The system can process multiple transactions in a single run, enabling scalable offline analysis or bulk categorization.

• Config-driven design allows rapid onboarding of new categories - Administrators can update or expand taxonomy categories directly through YAML configuration without modifying any code.

## 8. Running the Project

### 1. Install Dependencies

pip install -r requirements.txt

## 2. Train the Model

python main.py --mode train

### 3. Evaluate the Model

python main.py --mode evaluate

### 4. Start the Web Application

python app.py

### 5. start the entire server

python main.py --mode all --run-server

### Navigate to:

- <a href= "http://localhost:8787"> Link </a>

## 9. Demonstration Screenshots

### 1. Home / Landing Page

Shows the introductory interface where users access the system, understand its purpose, and begin the transaction categorization workflow.

<img width="1888" height="907" alt="Screenshot 2025-11-22 155806" src="https://github.com/user-attachments/assets/63b451d3-a187-4e9b-b98b-2feb49f97436" />

<img width="1894" height="920" alt="Screenshot 2025-11-22 155819" src="https://github.com/user-attachments/assets/75242b21-b092-4f38-8aaf-af4a1c8371a8" />

### 2. Categorization Form

Displays the input section where users enter raw transaction text to be analyzed and classified by the model.

<img width="1893" height="927" alt="Screenshot 2025-11-22 155832" src="https://github.com/user-attachments/assets/e76308d1-d8aa-4b3d-a918-c4141cdcc316" />

<img width="1852" height="918" alt="Screenshot 2025-11-22 155846" src="https://github.com/user-attachments/assets/43bd3d73-5c45-4547-ac3a-a09dfb702974" />

### 3. Prediction + Explainability + Feedback

Presents the model’s predicted category, confidence score, explanation chart, and a feedback form enabling users to correct results for continuous learning.

<img width="1738" height="925" alt="Screenshot 2025-11-22 155922" src="https://github.com/user-attachments/assets/311ef17c-45ad-4dc1-a3c6-0f0a45125c6c" />

<img width="1739" height="739" alt="Screenshot 2025-11-22 155934" src="https://github.com/user-attachments/assets/94aa332a-ba10-46c7-8705-9f30ffa21b73" />

## 10. Project Demo

### Demo Video Link

This section provides the recorded walkthrough of the complete system, demonstrating model training, evaluation, the web interface, prediction workflow, and the feedback loop in action. Add your final video link here once recorded.

## 11. Final Conclusion

This AI powered Transaction Categorisation System delivers a fully automated, explainable, and extensible solution for classifying financial transactions with high accuracy. By combining a custom taxonomy, a robust ML model, transparent explainability, and a human feedback learning loop, the system enables organizations to reduce dependency on external APIs, lower operational costs, and maintain complete control over their data. Through local processing, configurable design, and continuous improvement capabilities, the project provides a scalable and reliable foundation for financial analytics, budgeting tools, and enterprise-grade categorisation workflows.

# Automated AI-Based Financial Transaction Categorisation

This project implements an end to end, fully local AI system for automatically categorizing financial transactions using machine learning, configurable taxonomies, explainable AI techniques, and human feedback integration. It is designed to provide a cost-effective, privacy-focused alternative to third party categorization APIs.

## 1. Project Overview

This project solves these challenges by providing a complete, standalone ML-powered categorisation engine that runs entirely on local infrastructure, supports custom categories, integrates explainability, and continuously improves through user feedback.

Modern financial applications require fast and accurate categorisation of raw transaction text such as “Starbucks”, “Amazon Purchase”, or “Shell Gas”. Traditional approaches rely on external paid APIs, which introduce cost, latency, and data-privacy concerns.

The system includes:

• Machine Learning Model (Linear SVM + TF-IDF)

• Fully local inference (no external APIs)

• Customisable taxonomy using YAML

• Explainability using model coefficients

• Human feedback loop with CSV storage

• Web interface built with Flask

## 2. Technology Stack

• Python 3.10 – The core programming language used to implement the entire machine learning pipeline and web application.

• Scikit-learn – Provides the ML algorithms and tools for training, evaluating, and deploying the transaction categorization model.

• TF-IDF Vectorizer – Converts raw transaction text into numerical feature vectors used by the machine learning model.

• Linear Support Vector Machine (SVM) – The main classification algorithm that predicts transaction categories with high accuracy.

• Pandas & NumPy – Used for data processing, cleaning, transformation, and dataset manipulation.

• Flask – Powers the local web interface for running predictions, showing explanations, and collecting user feedback.

• YAML – Stores the editable taxonomy configuration, allowing categories to be updated without modifying code.

• Matplotlib – Generates evaluation artifacts including the confusion matrix for model assessment.

• Chart.js – Renders feature contribution and explainability charts in the browser UI.

• HTML/CSS/Bootstrap – Builds the responsive and modern frontend layout for the user interface.

## 3. System Architecture

• Dataset Generation and Ingestion - Collects raw or synthetic transaction strings and stores them locally to build a customizable training dataset.

• Preprocessing & Vectorization - Cleans input text and converts it into TF-IDF numerical vectors that the ML model can understand.

• Model Training - Uses a Linear SVM classifier to learn patterns and boundaries between transaction categories.

• Evaluation - Produces confusion matrices and F1-score reports to measure classification performance and ensure transparency.

• Explainability - Extracts feature importance signals from model coefficients to show why a prediction was made.

• Inference API - A Flask-based prediction endpoint that returns real time category and confidence scores.

• Local Web UI - A browser interface enabling users to categorize transactions, view explanations, and interact with the system.

• Human Feedback Loop - Stores user corrected labels into feedback.csv to continuously refine and improve the model.

• Retraining Capability - Allows new training sessions that incorporate user feedback, enabling adaptive and ever improving performance.

## 4. Data Model & Storage

• Processed Dataset – The cleaned and vector-ready transaction dataset is stored at data/processed/processed.csv for training and evaluation.

• Feedback Storage – User-submitted corrections are logged in data/feedback/feedback.csv to enable continuous model improvement.

• Taxonomy Definition – All category names and IDs are defined in configs/taxonomy.yaml, allowing full admin control without code changes.

• Model Artifacts – Trained ML models and vectorizers are saved in artifacts/models/ for reproducible inference and retraining.

• Evaluation Artifacts – Metrics such as confusion matrices and classification reports are stored in artifacts/metrics/ for transparent performance analysis.

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

## 6. Security and Compliance

• 100% local processing - All model operations run entirely on the user's device, ensuring full data ownership and eliminating external dependencies.

• No external APIs - The system does not send any transaction data to third party services, preventing API related privacy or security risks.

• No PII collected - The model handles generic transaction descriptions without storing or processing personally identifiable information.

• User feedback stored locally - Corrections provided by users are saved in a secure local CSV file, maintaining full control over learning data.

• Transparent explainability methods - Feature based explanations offer clear visibility into how each classification decision is made, supporting responsible AI practices.

## 7. Scalability and Performance

• Inference < 50ms locally - The model delivers near instant category predictions, enabling smooth real time usage even on modest hardware.

• Model size < 10 MB - A compact model footprint ensures fast loading, low memory usage, and easy deployment in local or embedded environments.

• Supports batch inference - The system can process multiple transactions in a single run, enabling scalable offline analysis or bulk categorization.

• Config-driven design allows rapid onboarding of new categories - Administrators can update or expand taxonomy categories directly through YAML configuration without modifying any code.

## 8. Running the Project

### 1. Install Dependencies

pip install -r requirements.txt

## 2. Train the Model

python main.py --mode train

### 3. Evaluate the Model

python main.py --mode evaluate

### 4. Start the Web Application

python app.py

### 5. start the entire server

python main.py --mode all --run-server

### Navigate to:

- <a href= "http://localhost:8787"> Link </a>

## 9. Demonstration Screenshots

### 1. Home / Landing Page

Shows the introductory interface where users access the system, understand its purpose, and begin the transaction categorization workflow.

<img width="1888" height="907" alt="Screenshot 2025-11-22 155806" src="https://github.com/user-attachments/assets/63b451d3-a187-4e9b-b98b-2feb49f97436" />

<img width="1894" height="920" alt="Screenshot 2025-11-22 155819" src="https://github.com/user-attachments/assets/75242b21-b092-4f38-8aaf-af4a1c8371a8" />

### 2. Categorization Form

Displays the input section where users enter raw transaction text to be analyzed and classified by the model.

<img width="1893" height="927" alt="Screenshot 2025-11-22 155832" src="https://github.com/user-attachments/assets/e76308d1-d8aa-4b3d-a918-c4141cdcc316" />

<img width="1852" height="918" alt="Screenshot 2025-11-22 155846" src="https://github.com/user-attachments/assets/43bd3d73-5c45-4547-ac3a-a09dfb702974" />

### 3. Prediction + Explainability + Feedback

Presents the model’s predicted category, confidence score, explanation chart, and a feedback form enabling users to correct results for continuous learning.

<img width="1738" height="925" alt="Screenshot 2025-11-22 155922" src="https://github.com/user-attachments/assets/311ef17c-45ad-4dc1-a3c6-0f0a45125c6c" />

<img width="1739" height="739" alt="Screenshot 2025-11-22 155934" src="https://github.com/user-attachments/assets/94aa332a-ba10-46c7-8705-9f30ffa21b73" />

### 4. Confusion matrix

Visualizing model performance across transaction categories True vs. Predicted classifications with accuracy metrics

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/ed1db030-de51-4a98-9b88-0c03e93293f7" />

## 10. Project Demo

### Demo Video Link

This section provides the recorded walkthrough of the complete system, demonstrating model training, evaluation, the web interface, prediction workflow, and the feedback loop in action. Add your final video link here once recorded.

## 11. Final Conclusion

This AI powered Transaction Categorisation System delivers a fully automated, explainable, and extensible solution for classifying financial transactions with high accuracy. By combining a custom taxonomy, a robust ML model, transparent explainability, and a human feedback learning loop, the system enables organizations to reduce dependency on external APIs, lower operational costs, and maintain complete control over their data. Through local processing, configurable design, and continuous improvement capabilities, the project provides a scalable and reliable foundation for financial analytics, budgeting tools, and enterprise-grade categorisation workflows.
