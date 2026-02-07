import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 – Classification Models")
st.write("Heart Disease Prediction using Multiple Classification Models")

# -------------------------------
# FILE UPLOAD (DEFINE FIRST!)
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)", type=["csv"]
)

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Target & Feature Processing
    # -------------------------------
    y = df["num"].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop("num", axis=1)
    X = X.drop(["id", "dataset"], axis=1)

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------------------------------
    # Model Selection
    # -------------------------------
    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    # -------------------------------
    # Initialize Model
    # -------------------------------
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(eval_metric="logloss", random_state=42)

    # -------------------------------
    # Train & Predict
    # -------------------------------
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # -------------------------------
    # Metrics
    # -------------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
    col2.metric("Precision", round(precision_score(y, y_pred), 3))
    col3.metric("Recall", round(recall_score(y, y_pred), 3))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y, y_pred), 3))
    col5.metric("AUC", round(roc_auc_score(y, y_proba), 3))
    col6.metric("MCC", round(matthews_corrcoef(y, y_pred), 3))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=11)

    st.pyplot(fig)
