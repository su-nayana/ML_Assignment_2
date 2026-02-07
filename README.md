# ML Assignment 2 – Classification Models & Streamlit App

## Problem Statement
The objective of this assignment is to implement multiple classification models on a single dataset, evaluate their performance using standard metrics, and deploy an interactive Streamlit web application to demonstrate the models. The assignment provides hands-on experience with an end-to-end machine learning workflow including data preprocessing, model training, evaluation, UI development, and deployment.

---

## Dataset Description
The Heart Disease UCI dataset is used for this assignment. The dataset contains patient health-related attributes such as age, sex, chest pain type, cholesterol levels, resting blood pressure, ECG results, and other clinical features.  

- Number of records: > 900  
- Number of features: 14  
- Target variable:  
  - `num = 0` → No heart disease  
  - `num > 0` → Presence of heart disease (converted to binary classification)

The dataset includes both numerical and categorical features and contains missing values, making it suitable for demonstrating preprocessing techniques.

---

## Models Implemented
The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest Classifier (Ensemble Model)  
6. XGBoost Classifier (Ensemble Model)

---

## Evaluation Metrics
Each model was evaluated using the following metrics:

- Accuracy  
- Area Under the ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

These metrics provide a comprehensive view of model performance, especially for binary classification problems.

---

## Model Performance Observations
- Ensemble models such as Random Forest and XGBoost achieved the best overall performance across most metrics.
- Logistic Regression and Naive Bayes provided strong baseline performance with good interpretability.
- KNN performed well but is sensitive to feature scaling.
- Decision Tree showed comparatively lower generalization performance, likely due to overfitting.
- MCC was particularly useful in evaluating balanced performance between positive and negative classes.

---

## Streamlit Web Application
An interactive Streamlit web application was developed with the following features:

- CSV file upload functionality  
- Dataset preview  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  

The app allows users to dynamically select a classification model and view its performance on the uploaded dataset.

---

## Repository Structure
ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── data/
│ └── heart_disease_uci.csv
└── model/
└── model_training.ipynb


---

## Deployment
The Streamlit application is deployed on Streamlit Community Cloud and can be accessed using the shared public link. The GitHub repository contains all necessary code, dataset, and dependency information required to reproduce the results.

---

## Conclusion
This assignment demonstrates an end-to-end machine learning workflow, from data preprocessing and model implementation to evaluation, visualization, and deployment. It provides practical exposure to real-world ML application development using Python and Streamlit.
