
import os
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

def index(request):
    # Load cluster data to pass to template or just let template fetch it?
    # Let's load it here for server-side rendering of the table if we want
    json_path = os.path.join(settings.BASE_DIR, 'api/static/api/assets/clusters_data.json')
    context = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            context['clusters'] = data
            
    return render(request, 'api/index.html', context)

def clusters_data(request):
    json_path = os.path.join(settings.BASE_DIR, 'api/static/api/assets/clusters_data.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return JsonResponse(data)
    return JsonResponse({'error': 'Data not found'}, status=404)
