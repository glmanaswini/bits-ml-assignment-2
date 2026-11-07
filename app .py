import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

st.set_page_config(page_title="Obesity Classification App", page_icon="📊", layout="wide")

st.title("M.Tech ML Assignment 2: Multi-Class Classification App")
st.markdown("**Dataset:** Obesity Risk Prediction | **Roll Number:** 2025NS04097")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])
model_display_name = st.sidebar.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Support Vector Machine"]
)

# Model key mapping
model_mapping = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm"
}
model_key = model_mapping[model_display_name]

# Load test dataset
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    try:
        test_df = pd.read_csv("test_data.csv")
    except FileNotFoundError:
        st.warning("Please upload 'test_data.csv' via the sidebar.")
        st.stop()

if 'NObeyesdad' in test_df.columns:
    X_test = test_df.drop(columns=['NObeyesdad'])
    y_test = test_df['NObeyesdad']
else:
    X_test = test_df
    y_test = None

# Load model & scaler from model/ folder
try:
    model = joblib.load(f"model/{model_key}.pkl")
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Error loading model from 'model/{model_key}.pkl': {e}")
    st.stop()

# Transform features for scaled models
if model_key in ["logistic_regression", "knn", "svm"]:
    X_test_processed = scaler.transform(X_test)
else:
    X_test_processed = X_test

y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed) if hasattr(model, "predict_proba") else None

st.subheader(f"Performance Metrics: {model_display_name}")

if y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted') if y_prob is not None else 0.0
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC Score", f"{mcc:.4f}")

    st.divider()
    st.subheader("Classification Report & Confusion Matrix")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(fig)
else:
    st.info("Target column 'NObeyesdad' not found in test dataset.")
    st.write(pd.DataFrame({"Predicted Label": y_pred}))
