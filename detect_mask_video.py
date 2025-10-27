# detect_mask_video.py
# Real-time Face Mask Detection (Fixed and Improved)

import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.getcwd()
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'mask_detector_model.h5')
IMG_SIZE = (128, 128)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print('❌ Model not found at:', MODEL_PATH)
    print('👉 Run train_mask_detector.py first to train and save the model.')
    raise SystemExit

# Load model
model = load_model(MODEL_PATH)
print('✅ Loaded model from:', MODEL_PATH)

# Load Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ Cannot open webcam. Try using VideoCapture(1) or (2).')
    raise SystemExit

print('🎥 Webcam started... Press "q" to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]

        try:
            face_resized = cv2.resize(face, IMG_SIZE)
        except:
            continue

        face_norm = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        # Predict mask
        pred = model.predict(face_input, verbose=0)[0][0]  # sigmoid output (0 to 1)
        confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100

        # ✅ Correct logic: if pred > 0.5 → Mask else No Mask
        if pred > 0.5:
            label = f"Mask ({confidence:.1f}%) 😷"
            color = (0, 255, 0)  # Green
        else:
            label = f"No Mask ({confidence:.1f}%) ❌"
            color = (0, 0, 255)  # Red

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('😷 Face Mask Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
