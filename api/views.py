
import os
import json
import base64
import io
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

# Lazy loading helper
def get_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def get_plot_base64(fig):
    plt = get_plt()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e293b')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def index(request):
    # Heavy imports moved inside to prevent OOM on worker boot
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.neighbors import KNeighborsClassifier
    
    plt = get_plt()

    # Default clusters
    n_clusters = int(request.GET.get('n_clusters', 6))
    
    # Load Data (Optimized: read only needed cols)
    base_dir = settings.BASE_DIR
    # Try different paths to be safe
    dataset_path = os.path.join(base_dir, 'datasets', 'datasets', 'creditcard.csv')
    if not os.path.exists(dataset_path):
         dataset_path = os.path.join(base_dir, 'datasets', 'creditcard.csv')
    
    context = {'n_clusters': n_clusters}
    
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path, usecols=['V10', 'V14', 'Class'])
            X = df[['V10', 'V14']].copy()
            
            # KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Stats
            counts = pd.Series(clusters).value_counts().sort_index().to_dict()
            context['counts'] = counts
            
            # 1. Scatter Plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            # Customizing for the glassmorphism look
            ax1.set_facecolor('#1e293b')
            fig1.patch.set_facecolor('#1e293b')
            
            scatter = ax1.scatter(X['V10'], X['V14'], c=clusters, cmap='viridis', marker='.', s=10, alpha=0.5)
            ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
            ax1.set_title(f'K-Means (k={n_clusters}) - V10 vs V14', color='white')
            ax1.set_xlabel('V10', color='white')
            ax1.set_ylabel('V14', color='white')
            ax1.legend()
            ax1.tick_params(colors='white')
            
            context['scatter_plot'] = get_plot_base64(fig1)

            # 2. Decision Boundary (Approximation with KNN)
            # Sample for speed if dataset is large (creditcard.csv is ~280k rows, sampling 50k is safe)
            sample_size = min(20000, len(X))
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[indices]
            y_sample = clusters[indices]
            
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_sample, y_sample)
            
            x_min, x_max = X['V10'].min() - 1, X['V10'].max() + 1
            y_min, y_max = X['V14'].min() - 1, X['V14'].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), # Lower res for speed
                                 np.arange(y_min, y_max, 0.2)) # 0.2 step
            
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            plt.style.use('dark_background')
            ax2.set_facecolor('#1e293b')
            fig2.patch.set_facecolor('#1e293b')
            
            ax2.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            ax2.scatter(X_sample['V10'], X_sample['V14'], c=y_sample, s=5, alpha=0.6, cmap='viridis', marker='.')
            ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
            ax2.set_title(f'Fronteras de Decisión (k={n_clusters})', color='white')
            ax2.set_xlabel('V10', color='white')
            ax2.set_ylabel('V14', color='white')
            ax2.tick_params(colors='white')
            
            context['boundary_plot'] = get_plot_base64(fig2)
            
        except Exception as e:
            context['error'] = str(e)
            print(f"Error generating plots: {e}")

    return render(request, 'api/index.html', context)

def clusters_data(request):
    # Keep this for API compatibility, but maybe just return static if needed or implement logic
    # For now, let's just return a placeholder or the static file if it exists
    json_path = os.path.join(settings.BASE_DIR, 'api/static/api/assets/clusters_data.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return JsonResponse(data)
    return JsonResponse({'error': 'Data not found'}, status=404)

def clusters_data(request):
    json_path = os.path.join(settings.BASE_DIR, 'api/static/api/assets/clusters_data.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return JsonResponse(data)
    return JsonResponse({'error': 'Data not found'}, status=404)
