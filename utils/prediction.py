import numpy as np

def predict_class(model, preprocessed_image):
    """
    Runs prediction using the selected model and preprocessed input.
    Returns class probabilities.
    """
    preds = model.predict(preprocessed_image)
    return preds[0] if preds.ndim > 1 else preds
