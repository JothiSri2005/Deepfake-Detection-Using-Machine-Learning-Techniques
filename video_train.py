# Video Training (for local system)
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, TimeDistributed, GRU, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Paths
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "IP")
REAL_VIDEOS_DIR = os.path.join(desktop_path, "SDFVD", "videos_real")
FAKE_VIDEOS_DIR = os.path.join(desktop_path, "SDFVD", "videos_fake")

# Parameters
IMG_SIZE = 224
FRAMES_PER_VIDEO = 10
BATCH_SIZE = 8

# Load files
real_files = [os.path.join(REAL_VIDEOS_DIR, f) for f in os.listdir(REAL_VIDEOS_DIR) if f.endswith('.mp4')]
fake_files = [os.path.join(FAKE_VIDEOS_DIR, f) for f in os.listdir(FAKE_VIDEOS_DIR) if f.endswith('.mp4')]

# Balance dataset
min_samples = min(len(real_files), len(fake_files))
real_files = real_files[:min_samples]
fake_files = fake_files[:min_samples]

train_files = real_files[:int(0.8 * min_samples)] + fake_files[:int(0.8 * min_samples)]
val_files = real_files[int(0.8 * min_samples):] + fake_files[int(0.8 * min_samples):]
train_labels = [0] * int(0.8 * min_samples) + [1] * int(0.8 * min_samples)
val_labels = [0] * (min_samples - int(0.8 * min_samples)) + [1] * (min_samples - int(0.8 * min_samples))

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames)

def create_dataset(files, labels):
    def generator():
        for i in range(len(files)):
            frames = extract_frames(files[i])
            yield frames, labels[i]
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(), dtype=tf.int32)))
    return dataset.shuffle(len(files)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_files, train_labels)
val_dataset = create_dataset(val_files, val_labels)

def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    inputs = Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3))
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = GRU(64, return_sequences=False, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

checkpoint_path = os.path.join(desktop_path, "deepfake_detector_best.h5")

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True)
]

model.fit(train_dataset, validation_data=val_dataset, epochs=30, callbacks=callbacks)
model.save(os.path.join(desktop_path, "deepfake_detector_video.keras"))
