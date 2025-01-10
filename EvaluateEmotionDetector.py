import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load model
emotion_model = load_model('model/emotion_model_full.h5')
print("Loaded model from disk")

# Data preprocessing
test_data_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False  # Required for correct confusion matrix
)

# Model prediction
predictions = emotion_model.predict(test_generator)

# Confusion matrix
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=list(emotion_dict.values()))
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))
