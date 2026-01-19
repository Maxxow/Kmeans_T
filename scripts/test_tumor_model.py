
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'tumor_model.joblib')

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    clf = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    
    # Create dummy input (64x64 flattened = 4096 features)
    dummy_input = np.random.randint(0, 255, size=(1, 4096))
    
    print("Predicting on dummy input...")
    prediction = clf.predict(dummy_input)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    test_model()
