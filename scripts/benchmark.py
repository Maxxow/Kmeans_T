
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import os

def benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'datasets', 'datasets', 'creditcard.csv')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, 'datasets', 'creditcard.csv')

    print("--- Starting Benchmark ---")
    
    # 1. Load Data
    start_time = time.time()
    df = pd.read_csv(dataset_path, usecols=['V10', 'V14', 'Class'])
    X = df[['V10', 'V14']].copy()
    load_time = time.time() - start_time
    print(f"Data Loading: {load_time:.4f} seconds")

    # 2. KMeans
    start_time = time.time()
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    kmeans_time = time.time() - start_time
    print(f"KMeans Training: {kmeans_time:.4f} seconds")

    # 3. Scatter Plot
    start_time = time.time()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(X['V10'], X['V14'], c=clusters, cmap='viridis', marker='.', s=10, alpha=0.5)
    ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)
    plt.close(fig1)
    scatter_time = time.time() - start_time
    print(f"Scatter Plot Generation: {scatter_time:.4f} seconds")

    # 4. Decision Boundary (KNN)
    start_time = time.time()
    sample_size = min(20000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[indices]
    y_sample = clusters[indices]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_sample, y_sample)
    
    x_min, x_max = X['V10'].min() - 1, X['V10'].max() + 1
    y_min, y_max = X['V14'].min() - 1, X['V14'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.close() # cleanup
    boundary_time = time.time() - start_time
    print(f"Decision Boundary (KNN + Grid Predict): {boundary_time:.4f} seconds")

    total_time = load_time + kmeans_time + scatter_time + boundary_time
    print(f"--- Total Theoretical Time: {total_time:.4f} seconds ---")

if __name__ == "__main__":
    benchmark()
