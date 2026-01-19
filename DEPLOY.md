
# Deployment Guide

This guide explains how to deploy the `fraud_detection` application to a free hosting service.

## Prerequisites
- **GitHub Account**: To host the code.
- **Render / Railway Account**: To host the live application.
- **Git Installed**.

## 1. Prepare Repository

Initialize a Git repository and push your code to GitHub.

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Fraud Detection App"

# Create a new repository on GitHub and link it
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 2. Deploy to Render (Free Tier)

Render is recommended for its simplicity and free web service tier.

1.  Log in to [Render Dashboard](https://dashboard.render.com/).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  Configure the service:
    - **Name**: `fraud-detection`
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
    - **Start Command**: `gunicorn fraud_detection.wsgi`
5.  Click **Create Web Service**.

## 3. Deploy to Railway (Alternative)

1.  Log in to [Railway](https://railway.app/).
2.  Click **New Project** -> **Deploy from GitHub repo**.
3.  Select your repository.
4.  Railway will automatically detect the `Procfile` and `requirements.txt`.
5.  It should deploy automatically.

## 4. Verify Model Existence

The trained model is located at `api/static/api/assets/kmeans_6_clusters.joblib`.
You can load it in your code using:
```python
import joblib
model = joblib.load('path/to/kmeans_6_clusters.joblib')
```
(Note: Since this app currently serves static assets, the model file is available as a static file downloaded to the server).

## Troubleshooting
- **Static Files 404**: Ensure `whitenoise` is configured in `MIDDLEWARE` and `STATICFILES_STORAGE`.
- **Port Error**: Render/Railway set the `$PORT` environment variable automatically; `gunicorn` respects this.
- **Static Files 404**: Ensure `whitenoise` is configured in `MIDDLEWARE` and `STATICFILES_STORAGE`.
- **Port Error**: Render/Railway set the `$PORT` environment variable automatically; `gunicorn` respects this.
- **Memory Usage**: The app is now fully static (images are pre-generated), so it uses very little RAM. It is safe for all free tiers.
