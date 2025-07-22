import numpy as np

def preprocess_image(image, target_size=(224, 224)):
    """
    Resizes and normalizes the image for model prediction.
    """
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:  # remove alpha channel if present
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)
