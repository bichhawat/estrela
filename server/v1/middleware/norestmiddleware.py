import requests
import json
from django.core import serializers
from django.http import JsonResponse

class NoRestMiddleware(object):
    def __init__(self, get_response):
        self.get_response = get_response
        self.norest_url = 'http://localhost:8001'

    def __call__(self, request):
        if not (request.path_info.startswith("/media/") or request.path_info.startswith("/admin/")):
            mutable = request.GET._mutable
            request.GET._mutable = True
            request.GET['format'] = 'json'
            request.GET._mutable = mutable
            response = self.get_response(request)
        else:
            return self.get_response(request)
        if request.method != 'GET':
            return response

        if request.path_info.startswith("/login"):
            return self.get_response(request)
                    
        url = self.norest_url + request.path_info
        new_response = requests.post(url, json={
            'norest_response': json.loads(response.content.decode('utf-8')), 
            'norest_user': serializers.serialize('json', [request.user]) if request.user.is_authenticated() else None,
            'norest_get': request.GET.__dict__,
            'norest_post': request.POST.__dict__
            })
        response.content = new_response.content
        response.content_type = "application/json"
        response['Content-Length'] = len(response.content)
        return response
