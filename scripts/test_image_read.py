
import matplotlib.image as mpimg
import os
import numpy as np

# Path to a known tiff file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(BASE_DIR, 'graficacion', 'tumors', 'NewMRI', 'Healthcare+AI+Datasets', 'Healthcare AI Datasets', 'Brain_MRI', 'TCGA_CS_5395_19981004', 'TCGA_CS_5395_19981004_1.tif')

try:
    print(f"Attempting to read {IMG_PATH}")
    img = mpimg.imread(IMG_PATH)
    print(f"Success! Image shape: {img.shape}")
except Exception as e:
    print(f"Failed to read image: {e}")
