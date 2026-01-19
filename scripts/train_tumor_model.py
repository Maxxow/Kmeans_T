
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'graficacion', 'tumors', 'NewMRI', 'Healthcare+AI+Datasets', 'Healthcare AI Datasets', 'Brain_MRI')
CSV_PATH = os.path.join(DATASET_PATH, 'data_mask.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'tumor_model.joblib')

IMAGE_SIZE = (64, 64)

def train_model():
    print(f"Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    images = []
    labels = []
    
    print("Preprocessing images (using Pillow)...")
    for index, row in df.iterrows():
        img_rel_path = row['image_path']
        img_full_path = os.path.join(DATASET_PATH, img_rel_path)
        
        if not os.path.exists(img_full_path):
            continue
            
        try:
            # Open image with Pillow
            with Image.open(img_full_path) as img:
                # Convert to grayscale
                img = img.convert('L') 
                # Resize
                img = img.resize(IMAGE_SIZE)
                # Convert to numpy array
                img_arr = np.array(img)
                # Flatten
                img_flat = img_arr.flatten()
                
                images.append(img_flat)
                labels.append(row['mask'])
            
        except Exception as e:
            # print(f"Error processing {img_full_path}: {e}")
            pass

    X = np.array(images)
    y = np.array(labels)
    
    print(f"Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
    
    if len(X) == 0:
        print("No images loaded. Check paths and Tif support.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    joblib.dump(clf, MODEL_SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    train_model()
