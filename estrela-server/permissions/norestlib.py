from django.http import HttpResponse
from django.core import serializers
from functools import wraps
import json


def norest_permission(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        request = args[0]
        body = json.loads(request.body)
        user = body['norest_user']
        user = list(serializers.deserialize('json', user))[
            0].object if user else None
        data = body['norest_response']
        get = body['norest_get']
        # print(get)
        get = get if get else None
        post = body['norest_post']
        post = post if post else None

        filtered_data = []
        storage = {}

        for obj in data:
            perm, default = func(request, storage, user, obj, get, post)
            # import pdb; pdb.set_trace()
            if perm:
                filtered_data.append(obj)
            elif default:
                for key in default:
                    obj[key] = default[key]
                filtered_data.append(obj)

        # print(filtered_data)
        return HttpResponse(json.dumps(filtered_data), content_type='application/json')
    return wrapper


def norest_view(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        request = args[0]
        body = json.loads(request.body)
        permissions = func(*args, **kwargs)
        filtered_data = []
        for permission in permissions:
            response = permission(request)
            filtered_data = json.loads(response.content)
            body['norest_response'] = filtered_data
            request._body = json.dumps(body)

        return HttpResponse(json.dumps(filtered_data), content_type='application/json')
    return wrapper