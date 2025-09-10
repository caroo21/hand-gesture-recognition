import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Konstanten
ROOT = "data"
CLASSES = ["fist", "hand_open", "peace", "thumbs_up"]
IMG_SIZE = 64  # Alle Bilder auf 64x64 Pixel verkleinern

def load_images_from_folder(folder_path, class_name, class_index):
    """lädt alle Bilder aus einem Ordner"""
    images = []
    labels = []

    print(f"Lade Bilder aus: {folder_path}")

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    for filename in image_files:
        filepath = os.path.join(folder_path, filename)

        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        #von BGR (OpenCV) zu RGB
        img_rgb= cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        images.append(img_normalized)
        labels.append(class_index)

    print(f"  → {len(images)} Bilder geladen für {class_name}")
    return images, labels

def load_all_data():
    """Lädt alle Trainings- und Validierungsdaten"""
    
    all_train_images = []
    all_train_labels = []
    all_val_images = []
    all_val_labels = []
    
    print("=== Lade alle Trainingsdaten ===")
    
    # Für jede Klasse (fist=0, hand_open=1, peace=2, thumbs_up=3)
    for class_index, class_name in enumerate(CLASSES):
        
        # Trainingsdaten laden
        train_folder = os.path.join(ROOT, "train", class_name)
        if os.path.exists(train_folder):
            train_imgs, train_lbls = load_images_from_folder(train_folder, class_name, class_index)
            all_train_images.extend(train_imgs)
            all_train_labels.extend(train_lbls)
        else:
            print(f"Warnung: Ordner {train_folder} nicht gefunden")
        
        # Validierungsdaten laden
        val_folder = os.path.join(ROOT, "val", class_name)
        if os.path.exists(val_folder):
            val_imgs, val_lbls = load_images_from_folder(val_folder, class_name, class_index)
            all_val_images.extend(val_imgs)
            all_val_labels.extend(val_lbls)
        else:
            print(f"Warnung: Ordner {val_folder} nicht gefunden")
    
    # Listen zu NumPy Arrays konvertieren
    X_train = np.array(all_train_images)
    y_train = np.array(all_train_labels)
    X_val = np.array(all_val_images)
    y_val = np.array(all_val_labels)
    
    print(f"Training: {X_train.shape[0]} Bilder, Shape: {X_train.shape}")
    print(f"Validierung: {X_val.shape[0]} Bilder, Shape: {X_val.shape}")
    print(f"Klassen: {len(CLASSES)} → {CLASSES}")
    
    return X_train, y_train, X_val, y_val

def show_sample_images(X_train, y_train):
    """Zeigt Beispielbilder aus jeder Klasse"""
    
    plt.figure(figsize=(12, 8))
    
    for class_idx in range(len(CLASSES)):
        # Erstes Bild dieser Klasse finden
        class_indices = np.where(y_train == class_idx)[0]
        
        if len(class_indices) > 0:
            img = X_train[class_indices[0]]
            
            plt.subplot(2, 2, class_idx + 1)
            plt.imshow(img)
            plt.title(f"Klasse {class_idx}: {CLASSES[class_idx]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Daten laden
    X_train, y_train, X_val, y_val = load_all_data()
    
    # Beispielbilder anzeigen
    if len(X_train) > 0:
        show_sample_images(X_train, y_train)
    else:
        print("Keine Trainingsdaten gefunden!")
