# train_mask_detector.py
# Train a simple CNN for binary face mask detection.
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

PROJECT_ROOT = os.getcwd()
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')  # expects dataset/with_mask and dataset/without_mask
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 15

print('Dataset directory:', DATASET_DIR)
print('Model directory:', MODEL_DIR)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Simple CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_path = os.path.join(MODEL_DIR, 'mask_detector_model.h5')
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    callbacks=[checkpoint, earlystop]
)

print('Training finished. Best model saved to:', checkpoint_path)

# Save training curves
import matplotlib.pyplot as plt
plt.figure(); plt.plot(history.history['loss'], label='train_loss'); plt.plot(history.history['val_loss'], label='val_loss'); plt.legend(); plt.savefig(os.path.join(MODEL_DIR,'loss.png'))
plt.figure(); plt.plot(history.history['accuracy'], label='train_acc'); plt.plot(history.history['val_accuracy'], label='val_acc'); plt.legend(); plt.savefig(os.path.join(MODEL_DIR,'acc.png'))
print('Saved acc/loss plots to model/')
