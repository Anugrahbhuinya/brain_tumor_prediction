import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

from utils.preprocessing import preprocess_image
from utils.model_loader import load_selected_model
from utils.prediction import predict_class

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Set page configuration
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

# Class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Model options and paths
MODEL_PATHS = {
    "Custom CNN": "models/custom_model_best.h5",
    "ResNet50": "models/brain_tumor_resnet50_best.h5",
    "DenseNet121": "models/brain_tumor_densenet_best.h5",
    "EfficientNetB0": "models/custom_model.h5"
}

# Sidebar - Model Selection
st.sidebar.title(" Settings")
model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))

# Main App Title
st.title(" Brain Tumor MRI Classifier")
st.markdown("Upload an MRI scan and select a model to classify the brain tumor.")

# Upload MRI Image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# Process Uploaded Image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    if st.button(" Predict"):
        with st.spinner("Loading model and predicting..."):
            model = load_selected_model(model_name, MODEL_PATHS)
            input_image = preprocess_image(image)
            preds = predict_class(model, input_image)
            predicted_class = class_labels[np.argmax(preds)]

        # Show prediction
        st.success(f" Predicted Tumor Type: **{predicted_class}**")
        st.subheader(" Prediction Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"- **{label}**: {preds[i]*100:.2f}%")

# Accuracy Comparison Section
st.markdown("---")
st.subheader(" Model Performance Comparison")

metrics_data = {
    "Model": ["Custom CNN", "ResNet50", "DenseNet121", "EfficientNetB0"],
    "Accuracy": [0.78, 0.79, 0.79, 0.71],
    "Precision": [0.80, 0.87, 0.86, 0.74],
    "Recall": [0.78, 0.86, 0.87, 0.71],
    "F1-Score": [0.77, 0.86, 0.86, 0.70]
}

df = pd.DataFrame(metrics_data)
st.dataframe(df.set_index("Model"))

# Bar Chart for Accuracy
st.markdown("###  Accuracy Comparison")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="Model", y="Accuracy", palette="Blues_d")
plt.ylim(0, 1)
plt.ylabel("Accuracy Score")
plt.title("Model Accuracy Comparison")
st.pyplot(fig)

# Confusion Matrix Section (optional/test-only)
st.markdown("---")
st.subheader(" Confusion Matrix and Classification Report")
if st.checkbox("Show Confusion Matrix (from test set)"):
    try:
        # Load or simulate y_true and y_pred if available
        y_true = np.load("data/y_true.npy")
        y_pred = np.load("data/y_pred.npy")

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        st.pyplot(fig)

        st.markdown("###  Classification Report")
        st.text(classification_report(y_true, y_pred, target_names=class_labels))
    except Exception as e:
        st.warning("❗ Please ensure `y_true.npy` and `y_pred.npy` are available in the data/ folder.")
        st.error(str(e))
