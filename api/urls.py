
from django.urls import path
from . import views

urlpatterns = [
    path('', views.tumor_view, name='index'), # Changed default to tumor_view
    path('fraud/', views.index, name='fraud_index'), 
    path('emotion/', views.emotion_view, name='emotion_index'),
    path('api/data', views.clusters_data, name='clusters_data'),
    path('api/predict_tumor', views.predict_tumor, name='predict_tumor'),
    path('api/predict_emotion', views.predict_emotion, name='predict_emotion'),
]
