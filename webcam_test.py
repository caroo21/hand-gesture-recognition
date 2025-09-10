import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Webcam konnte nicht geöffnet werden")
    exit()

print("Webcam erfolgreich geöffnet!")
print("Drücke 'q' zum Beenden")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fehler beim Lesen des Frames")
        break

    cv2.imshow('Webcam Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam geschlossen")