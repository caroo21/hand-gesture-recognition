import tensorflow as tf
import numpy as np
#from simple_model import load_limited_data
from prepare_data import load_all_data

# Model laden
model = tf.keras.models.load_model('hand_gesture_model.h5') #hand_gesture_model.h5 quick_hand_model.h5

# Testdaten laden
X_train, y_train, X_val, y_val = load_all_data()

print("Model Debug")

# Overall Evaluation
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nOverall Validation Accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
print(f"Overall Validation Loss: {val_loss:.3f}")

# Vorhersagen auf Validierungsdaten
predictions = model.predict(X_val)
predicted_classes = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1)

# Verteilung der Vorhersagen
classes = ["fist", "hand_open", "peace", "thumbs_up"]
print("\nVorhersage-Verteilung:")
for i, class_name in enumerate(classes):
    count = np.sum(predicted_classes == i)
    percentage = (count / len(predicted_classes)) * 100
    print(f"{class_name}: {count}/{len(predicted_classes)} ({percentage:.1f}%)")

# Echte Labels vs Vorhersagen
print("\nEchte Labels vs Vorhersagen:")
for i in range(min(20, len(X_val))):  # Erste 20 Beispiele
    true_class = classes[y_val[i]]
    pred_class = classes[predicted_classes[i]]
    confidence = predictions[i][predicted_classes[i]]
    print(f"Echt: {true_class:10} | Vorhergesagt: {pred_class:10} | Confidence: {confidence:.3f}")

# Accuracy pro Klasse
print("\nAccuracy pro Klasse:")
for i, class_name in enumerate(classes):
    class_mask = (y_val == i)
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(predicted_classes[class_mask] == i)
        print(f"{class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")

# Confidence Analyse 
print(f"\nConfidence Statistics:")
print(f"Mean Confidence: {np.mean(confidence_scores):.3f}")
print(f"Min Confidence: {np.min(confidence_scores):.3f}")
print(f"Max Confidence: {np.max(confidence_scores):.3f}")
