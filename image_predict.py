# Image Test (for local system)
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Define local path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "IP")
model_path = os.path.join(desktop_path, "deepfake_detector.h5")

# Load model
model = load_model(model_path)

# Prediction function
def predict_image(path):
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)[0][0]
        label = "Fake" if prediction > 0.5 else "Real"
        confidence = f"{prediction if label == 'Fake' else 1 - prediction:.2%}"
        return f"Prediction: {label} ({confidence})"
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    img_path = input("📷 Enter path to the image file: ").strip()
    result = predict_image(img_path)
    print(result)
