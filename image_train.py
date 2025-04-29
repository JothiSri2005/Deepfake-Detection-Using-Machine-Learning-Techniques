# For image training the model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import Counter
import os

# ✅ Set paths to dataset on local Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "IP")
train_dir = os.path.join(desktop_path, "IP PROJECT", "dataset", "train")
valid_dir = os.path.join(desktop_path, "IP PROJECT", "dataset", "valid")

# ✅ Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Debug class indices
print("Class Indices:", train_generator.class_indices)
print("Training Class Distribution:", Counter(train_generator.classes))

# Load EfficientNetV2B0
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze last 20 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Add classifier
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_generator, validation_data=valid_generator, epochs=15)

# Evaluate
val_loss, val_acc = model.evaluate(valid_generator)
print(f"✅ Validation Accuracy: {val_acc:.2%}")

# ✅ Save the model to Desktop
model_path = os.path.join(desktop_path, "IP PROJECT", "deepfake_detector.h5")
model.save(model_path)
print(f"✅ Training complete. Model saved at: {model_path}")
