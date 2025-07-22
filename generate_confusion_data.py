# generate_confusion_data.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Constants
model_path = "models/brain_tumor_resnet50_best.h5"  # or any other model
test_dir = r"C:\Users\ASUS\Desktop\brain_tumor_data\test"  # e.g. 'brain_tumor_data/test'
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
img_size = (224, 224)

# Load model
model = load_model(model_path)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save
np.save("data/y_true.npy", y_true)
np.save("data/y_pred.npy", y_pred)

# Optional: print classification report
print(classification_report(y_true, y_pred, target_names=class_labels))
