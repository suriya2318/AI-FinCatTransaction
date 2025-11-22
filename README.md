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


