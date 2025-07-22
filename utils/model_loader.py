import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_selected_model(model_name, model_paths):
    """
    Loads and caches the selected Keras model.
    """
    model_path = model_paths[model_name]
    return tf.keras.models.load_model(model_path)
