
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def generate_assets():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'datasets', 'datasets', 'creditcard.csv')
    
    # Fallback path check
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, 'datasets', 'creditcard.csv')
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found at {dataset_path}")
            return

    output_dir = os.path.join(base_dir, 'api', 'static', 'api', 'assets')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    X = df[['V10', 'V14']].copy()
    
    # KMeans with 6 clusters
    print("Training KMeans (6 clusters)...")
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Save Metadata
    metadata = {
        'n_clusters': 6,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'counts': pd.Series(clusters).value_counts().to_dict()
    }
    with open(os.path.join(output_dir, 'clusters_data.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved.")

    # Save Model
    print("Saving Model...")
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_6_clusters.joblib'))
    print("Model saved to kmeans_6_clusters.joblib")

    # 1. Scatter Plot
    print("Generating Scatter Plot...")
    plt.figure(figsize=(10, 6))
    # Use a dark style for the plot to match the UI reqs if possible, or just standard
    plt.style.use('dark_background')
    plt.scatter(X['V10'], X['V14'], c=clusters, cmap='viridis', marker='.', s=10, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title('K-Means Clustering (6 Clusters) - V10 vs V14')
    plt.xlabel('V10')
    plt.ylabel('V14')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter.png'))
    plt.close()

    # 2. Decision Boundary Plot
    # Using KNN to approximate decision boundary for visualization
    print("Generating Decision Boundary Plot...")
    knn = KNeighborsClassifier(n_neighbors=5)
    # Sampling for speed in KNN fitting/plotting if dataset is huge
    sample_size = min(50000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[indices]
    y_sample = clusters[indices]
    
    knn.fit(X_sample, y_sample)

    x_min, x_max = X['V10'].min() - 1, X['V10'].max() + 1
    y_min, y_max = X['V14'].min() - 1, X['V14'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X_sample['V10'], X_sample['V14'], c=y_sample, s=5, alpha=0.6, cmap='viridis', marker='.')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title('K-Means Decision Boundaries (6 Clusters)')
    plt.xlabel('V10')
    plt.ylabel('V14')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'decision_boundary.png'))
    plt.close()
    
    print("All assets generated successfully.")

if __name__ == "__main__":
    generate_assets()
