import os
import time
import cv2

ROOT = "data"
CLASSES = ["fist", "hand_open", "peace", "thumbs_up"]
SPLITS = {"train": 32, "val": 8}

os.makedirs(ROOT, exist_ok=True)
for split in SPLITS:
    for c in CLASSES:
        os.makedirs(os.path.join(ROOT, split, c), exist_ok=True)


def collect_images_for_class(class_name, split, num_images):
    """Sammelt Bilder für bestimmte Klasse und Split"""

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Fehler: Webcam konnte nicht geöffnet werden")
        return
    
    print(f"\n=== Sammle {num_images} Bilder für '{class_name}' ({split}) ===")
    images_taken = 0
    save_path = os.path.join(ROOT, split, class_name)
    
    while images_taken < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Klasse: {class_name} {split}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display_frame, f"Bilder: {images_taken}/{num_images}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "LEERTASTE = Foto, Q = Quit", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Datensammlung', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): # Leertaste gedrückt
            filename = f"{class_name}_{split}_{images_taken:03d}.jpg"
            filepath = os.path.join(save_path, filename)
            cv2.imwrite(filepath, frame)
            
            print(f"Bild gespeichert: {filename}")
            images_taken += 1
            time.sleep(0.5)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Verfügbare Klassen:", CLASSES)

    #welche Klasse soll gesammelt werden
    class_to_collect = CLASSES[3]
    collect_images_for_class(class_to_collect, "train", SPLITS["train"])

    input("\nDrücke Enter um mit Validierungsdaten fortzufahren...")

    collect_images_for_class(class_to_collect, "val", SPLITS["val"])