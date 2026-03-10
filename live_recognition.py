import cv2 
import numpy as np 
import tensorflow as tf 
import time
 
CLASSES = ["fist", "hand_open", "peace", "thumbs_up"]
IMG_SIZE = 64
MODEL_PATH = 'hand_gesture_model.keras'

def preprocess_frame(frame):
    """Bereitet ein Webcam-Frame für das Model vor"""

    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Batch-Dimension hinzufügen: (64,64,3) → (1,64,64,3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def live_recognition():
    """Startet Live-Erkennung mit Webcam"""
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    print("Drücke q zum Beenden")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)

        predictions = model.predict(processed_frame, verbose = 0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = CLASSES[predicted_class_idx]
        display_frame = frame.copy()
        
        # Haupttext (große Schrift)
        cv2.putText(display_frame, f"Geste: {predicted_class}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Confidence (kleinere Schrift)
        cv2.putText(display_frame, f"Sicherheit: {confidence:.2f}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Alle Wahrscheinlichkeiten anzeigen
        y_pos = 150
        for i, class_name in enumerate(CLASSES):
            prob = predictions[0][i]
            color = (0, 255, 0) if i == predicted_class_idx else (255, 255, 255)
            cv2.putText(display_frame, f"{class_name}: {prob:.3f}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30
        
        # Anleitung
        cv2.putText(display_frame, "Druecke 'q' zum Beenden", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Handgesten Erkennung', display_frame)
        
        # 'q' zum Beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def debug_live_frame(frame, model):
    """Debuggt ein Live-Frame"""
    
    # Preprocessing wie gewohnt
    processed_frame = preprocess_frame(frame)
    
    # Vorhersage
    predictions = model.predict(processed_frame, verbose=0)
    
    # Debug-Info
    print("\n=== FRAME DEBUG ===")
    print(f"Original Frame Shape: {frame.shape}")
    print(f"Processed Frame Shape: {processed_frame.shape}")
    print(f"Processed Frame Min/Max: {processed_frame.min():.3f} / {processed_frame.max():.3f}")
    
    print("\nAlle Vorhersagen:")
    for i, class_name in enumerate(CLASSES):
        prob = predictions[0][i]
        print(f"  {class_name}: {prob:.4f}")
    
    # Zeige verarbeitetes Bild
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Original Webcam")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_frame[0])  # Batch-Dimension entfernen
    plt.title("Verarbeitet für Model")
    plt.axis('off')
    
    plt.show()


if __name__ == "__main__":
    live_recognition()