import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Konstanten - HIER KANNST DU ANPASSEN
ROOT = "data"
CLASSES = ["fist", "hand_open", "peace", "thumbs_up"]
IMG_SIZE = 64
EPOCHS = 15  # Weniger Epochen = schneller
BATCH_SIZE = 8  # Kleinere Batches = weniger RAM

# Weniger Bilder verwenden für schnelleres Training
MAX_IMAGES_PER_CLASS = 20  # Statt alle 64, nur 20 pro Klasse verwenden

def load_limited_data():
    all_images = []
    all_labels = []

    for class_index, class_name in enumerate(CLASSES):
        class_images = []
        
        for split in ["train", "val"]:
            folder = os.path.join(ROOT, split, class_name)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
                
                for filename in files[:MAX_IMAGES_PER_CLASS//2]:  # Hälftig aufteilen
                    filepath = os.path.join(folder, filename)
                    img = cv2.imread(filepath)
                    
                    if img is not None:
                        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        img_normalized = img_rgb.astype(np.float32) / 255.0
                        
                        class_images.append(img_normalized)
        class_images = class_images[:MAX_IMAGES_PER_CLASS]
        labels = [class_index] * len(class_images)
        
        all_images.extend(class_images)
        all_labels.extend(labels)
        print(f"  {class_name}: {len(class_images)} Bilder")

    X = np.array(all_images)
    y = np.array(all_labels)

    # NEUE ZEILEN: Daten mischen
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Einfacher Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training: {len(X_train)}, Validierung: {len(X_val)}")
    return X_train, y_train, X_val, y_val

def create_simple_model():
    """Kleineres, schnelleres Model"""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Weniger Layer = schneller
        layers.Conv2D(16, (3, 3), activation='relu'),  # Weniger Filter
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(32, activation='relu'),  # Kleinere Dense Layer
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("=== Schnelles Handgesten-Training ===")
    
    # Daten laden
    X_train, y_train, X_val, y_val = load_limited_data()
    
    # Model erstellen
    model = create_simple_model()
    model.summary()
    
    # Schnelles Training
    print(f"\n=== Training für {EPOCHS} Epochen ===")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Model speichern
    model.save('quick_hand_model.h5')
    print("\n✅ Schnelles Training abgeschlossen!")
    print("Model gespeichert als 'quick_hand_model.h5'")