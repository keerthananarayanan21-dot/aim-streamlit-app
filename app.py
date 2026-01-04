import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Earnings Manipulation Detection", layout="wide")

st.title("Earnings Manipulation Detection App")
st.write("Machine Learning based detection using Beneish ratios")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_excel("Earnings Manipulator (1).xlsx")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# FEATURES & TARGET
# ---------------------------
features = [
    'DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI'
]

target = 'Manipulator'

X = df[features]
y = df[target].map({'Yes': 1, 'No': 0})

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# SCALING
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# LOAD BEST MODEL
# ---------------------------
model = joblib.load("best_earnings_manipulation_model.pkl")

# ---------------------------
# MODEL EVALUATION
# ---------------------------
if st.button("Evaluate Model"):
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
    col2.metric("Precision", round(precision_score(y_test, y_pred), 3))
    col3.metric("Recall", round(recall_score(y_test, y_pred), 3))
    col4.metric("F1 Score", round(f1_score(y_test, y_pred), 3))
    col5.metric("ROC AUC", round(roc_auc_score(y_test, y_prob), 3))

# ---------------------------
# USER INPUT PREDICTION
# ---------------------------
st.subheader("Check a Company Manually")

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        f"{feature}",
        value=float(df[feature].mean())
    )

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button("Predict Manipulation Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Manipulation (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Manipulation (Probability: {probability:.2%})")
