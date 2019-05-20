import requests
import json
from django.core import serializers
from django.http import JsonResponse
from v1.posts.models import post

class ExposeRequest(object):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        post.exposed_request = request
        response = self.get_response(request)
        return response
