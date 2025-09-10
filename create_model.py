import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from prepare_data import load_all_data

IMG_SIZE = 64
NUM_CLASSES = 4
CLASSES = ["fist", "hand_open", "peace", "thumbs_up"]
EPOCHS = 100  # Anzahl Trainingsdurchläufe
BATCH_SIZE = 32  # Bilder pro Trainingsschritt

def create_cnn_model():
    """Erstellt einfaches CNN-Model für Handgestenerkennung"""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE,IMG_SIZE, 3)),

        # erster convolutional Block - erkennt einfache Features,
        # Kanten, Linien, Bild wird kleiner, 
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2,2)), # halbiert Bildgröße

        # zweiter convolutional Block - 64 Filter erkennt mehr Features
        # Anzahl Parameter (3*3*32 + 1)*64
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)), # halbiert Bildgröße

        # Dritter Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        #Flatten für Dense Layer - von 6x6x64 zu 1D 2304
        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), #Verhindert overfitting

        #Output - 4 Neuronen für 4 Klassen
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model 

def create_better_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)),

        # Block 1 - mit Batch Normalization
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(), 
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    
        # Block 2 - mehr Filter
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    
        # Block 3 - noch mehr Filter
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling statt Flatten
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

def train_model():
    """Trainiert das Model mit gesammelten Daten"""

    # lade Daten
    X_train, y_train, X_val, y_val = load_all_data()
    print("X_train range:", X_train.min(), X_train.max())  
    if len(X_train) == 0:
        print("Fehler: Keine Trainingsdaten gefunden!")
        return None
    model = create_cnn_model()

    # Verschiedene Werte testen
    optimizer = keras.optimizers.Adam(learning_rate=0.001)   # Standard
    # optimizer = keras.optimizers.Adam(learning_rate=0.0001) # Langsamer
    # optimizer = keras.optimizers.Adam(learning_rate=0.01)   # Schneller

    # Callbacks definieren
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',     # Überwache Validation Accuracy
        patience=15,                # Stoppe nach 15 Epochen ohne Verbesserung
        restore_best_weights=True,  # Verwende beste Gewichte
        verbose=1
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Halbiere LR bei Stagnation
        patience=7,        # Nach 7 Epochen ohne Verbesserung
        min_lr=1e-6,
        verbose=1
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )

    # Training starten
    history = model.fit(
        X_train, y_train,
        batch_size= BATCH_SIZE,
        epochs=EPOCHS, # hoch, stoppt durch Early Stopping
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler],
        verbose = 1 #Zeige Fortschritt an
    )

    return model, history


if __name__=="__main__":

    model, history = train_model()

    model.save('hand_gesture_model.h5')